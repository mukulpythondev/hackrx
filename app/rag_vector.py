import os
import asyncio
import hashlib
import tempfile
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import logging
import time
from datetime import datetime

# Core imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import aiohttp
from pinecone import Pinecone, ServerlessSpec

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class OptimizedVectorRAG:
    def __init__(self):
        try:
            # Initialize Pinecone
            self.pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
            self.index_name = os.environ["PINECONE_INDEX_NAME"]
            dimension = 3072  # text-embedding-3-large dimension
            cloud = "aws"
            region = "us-east-1"

            # Check if index exists, else create
            existing_indexes = [i["name"] for i in self.pinecone.list_indexes()]
            if self.index_name not in existing_indexes:
                self.pinecone.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=cloud, region=region)
                )
                logger.info(f"✅ Created Pinecone index '{self.index_name}'")
                time.sleep(10)  # Wait for index to be ready

            self.index = self.pinecone.Index(self.index_name)

            # Initialize embeddings and LLM
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",
                chunk_size=1000  # Process embeddings in batches
            )
            self.llm = ChatOpenAI(
                model="gpt-4o",  # Faster model for better speed
                temperature=0,
                max_tokens=1000  # Limit response length for speed
            )

            logger.info("✅ Optimized Vector RAG system initialized")

        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise

    def generate_doc_id(self, pdf_url: str) -> str:
        """Generate consistent document ID from URL"""
        return hashlib.md5(pdf_url.encode()).hexdigest()[:12]

    def create_namespace(self, doc_id: str) -> str:
        """Create namespace for document isolation"""
        return f"doc_{doc_id}"

    async def download_pdf(self, url: str) -> str:
        """Download PDF to temp file with timeout"""
        try:
            timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to download PDF: HTTP {resp.status}")
                    content = await resp.read()
                    
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(content)
                logger.info(f"📥 Downloaded PDF ({len(content)} bytes) to {tmp.name}")
                return tmp.name
                
        except Exception as e:
            logger.error(f"❌ PDF download failed: {e}")
            raise

    def extract_and_chunk_pdf(self, temp_file: str, doc_id: str) -> List[Document]:
        """Extract text and create optimized chunks"""
        try:
            loader = PyPDFLoader(temp_file)
            docs = loader.load()
            
            if not docs:
                raise Exception("No content extracted from PDF")
            
            # Optimized text splitter for better chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Larger chunks for better context
                chunk_overlap=100,  # More overlap for continuity
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            chunks = splitter.split_documents(docs)
            
            # Add comprehensive metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'doc_id': doc_id,
                    'chunk_id': f"{doc_id}_{i:03d}",  # Padded for sorting
                    'chunk_index': i,
                    'page': chunk.metadata.get('page', 0),
                    'total_chunks': len(chunks),
                    'created_at': datetime.now().isoformat(),
                    'namespace': self.create_namespace(doc_id)
                })
            
            logger.info(f"📄 Extracted {len(chunks)} optimized chunks from PDF")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ PDF extraction failed: {e}")
            raise

    async def check_document_exists(self, doc_id: str) -> bool:
        """Check if document exists using namespace"""
        try:
            namespace = self.create_namespace(doc_id)
            
            # Query with namespace filter to check existence
            dummy_vector = [0.0] * 3072
            results = self.index.query(
                vector=dummy_vector,
                top_k=1,
                filter={'doc_id': doc_id},
                include_metadata=True,
                namespace=namespace
            )
            
            exists = len(results.matches) > 0
            if exists:
                logger.info(f"📋 Document {doc_id} already exists in namespace {namespace}")
            return exists
            
        except Exception as e:
            logger.warning(f"⚠️ Error checking document existence: {e}")
            return False

    async def store_vectors_optimized(self, chunks: List[Document], doc_id: str, batch_size: int = 100):
        """Store embeddings in Pinecone with namespace isolation"""
        try:
            namespace = self.create_namespace(doc_id)
            
            # Generate embeddings in batches for speed
            texts = [chunk.page_content for chunk in chunks]
            
            start_time = time.time()
            logger.info(f"🔄 Generating embeddings for {len(texts)} chunks...")
            
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None, self.embeddings.embed_documents, texts
            )
            
            embed_time = time.time() - start_time
            logger.info(f"✅ Generated embeddings in {embed_time:.2f}s")
            
            # Prepare vectors for upsert
            vectors_to_upsert = []
            for chunk, embedding in zip(chunks, embeddings):
                vectors_to_upsert.append({
                    "id": chunk.metadata["chunk_id"],
                    "values": embedding,
                    "metadata": {
                        "text": chunk.page_content[:1000],  # Pinecone metadata limit
                        "doc_id": chunk.metadata["doc_id"],
                        "chunk_index": chunk.metadata["chunk_index"],
                        "page": chunk.metadata.get("page", 0),
                        "total_chunks": chunk.metadata["total_chunks"],
                        "created_at": chunk.metadata["created_at"]
                    }
                })

            # Batch upsert with namespace
            upsert_start = time.time()
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                try:
                    self.index.upsert(
                        vectors=batch,
                        namespace=namespace
                    )
                    logger.info(f"📤 Upserted batch {i // batch_size + 1} ({len(batch)} vectors) to namespace {namespace}")
                except Exception as e:
                    logger.error(f"🛑 Batch upsert failed: {e}")
                    # Retry smaller batches
                    for vector in batch:
                        try:
                            self.index.upsert(vectors=[vector], namespace=namespace)
                        except Exception as retry_error:
                            logger.error(f"Failed to upsert single vector: {retry_error}")
            
            upsert_time = time.time() - upsert_start
            logger.info(f"✅ Stored {len(vectors_to_upsert)} vectors in {upsert_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Vector storage failed: {e}")
            raise

    async def vector_search(self, query: str, doc_id: str, top_k: int = 5) -> List[Dict]:
        """Optimized vector search with namespace isolation"""
        try:
            namespace = self.create_namespace(doc_id)
            
            # Generate query embedding
            query_embedding = await asyncio.get_event_loop().run_in_executor(
                None, self.embeddings.embed_query, query
            )
            
            # Search in specific namespace only
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter={'doc_id': doc_id},
                include_metadata=True,
                namespace=namespace
            )
            
            search_results = [
                {
                    'chunk_id': match.id,
                    'text': match.metadata['text'],
                    'score': match.score,
                    'chunk_index': match.metadata.get('chunk_index', 0),
                    'page': match.metadata.get('page', 0)
                }
                for match in results.matches
            ]
            
            scores = [f"{r['score']:.3f}" for r in search_results]
            logger.info(f"🔍 Vector search found {len(search_results)} chunks (scores: {scores})")
            return search_results
            
        except Exception as e:
            logger.error(f"❌ Vector search failed: {e}")
            return []

    async def generate_answer_optimized(self, query: str, context_docs: List[Dict]) -> str:
        """Generate answer with optimized prompt"""
        try:
            if not context_docs:
                return "No relevant information found in the document."
            
            # Sort by relevance score and take top chunks
            sorted_docs = sorted(context_docs, key=lambda x: x['score'], reverse=True)[:3]
            
            context = "\n\n".join([
                f"[Chunk {doc['chunk_index']} - Page {doc['page']} - Score: {doc['score']:.3f}]\n{doc['text']}" 
                for doc in sorted_docs
            ])
            
            # Optimized prompt for speed and accuracy
            prompt = f"""Based on the policy document context below, provide a precise answer to the question. 

INSTRUCTIONS:
- Give specific details (numbers, percentages, timeframes) when available
- If information is not in the context, state "Information not found in the provided document sections"
- Be concise but complete
- Quote exact terms when relevant

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
            
            start_time = time.time()
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.llm.invoke, prompt
            )
            
            generation_time = time.time() - start_time
            logger.info(f"💬 Answer generated in {generation_time:.2f}s")
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"❌ Answer generation failed: {e}")
            return f"Error generating answer: {str(e)}"

    async def process_document(self, pdf_url: str, doc_id: Optional[str] = None, force_reprocess: bool = False) -> str:
        """Process PDF with document isolation and caching"""
        if not doc_id:
            doc_id = self.generate_doc_id(pdf_url)
        
        # Check if document already exists
        if not force_reprocess and await self.check_document_exists(doc_id):
            logger.info(f"📋 Document {doc_id} already processed, skipping...")
            return doc_id
        
        temp_file = None
        try:
            start_time = time.time()
            logger.info(f"🚀 Processing new document: {pdf_url}")
            
            # Download PDF
            temp_file = await self.download_pdf(pdf_url)
            
            # Extract and chunk
            chunks = self.extract_and_chunk_pdf(temp_file, doc_id)
            
            if not chunks:
                raise Exception("No chunks extracted from PDF")
            
            # Store vectors with namespace isolation
            await self.store_vectors_optimized(chunks, doc_id)
            
            total_time = time.time() - start_time
            logger.info(f"✅ Document processed successfully: {doc_id} in {total_time:.2f}s")
            return doc_id
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            raise
            
        finally:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)

    async def query(self, question: str, doc_id: str) -> str:
        """Main query interface with optimized vector search"""
        try:
            start_time = time.time()
            logger.info(f"❓ Processing query: {question}")
            
            # Vector search only
            search_results = await self.vector_search(question, doc_id, top_k=5)
            
            # Generate answer
            answer = await self.generate_answer_optimized(question, search_results)
            
            elapsed_time = time.time() - start_time
            logger.info(f"⏱️ Query processed in {elapsed_time:.2f} seconds")
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ Query error: {e}")
            return f"Error processing query: {str(e)}"

    async def batch_query(self, questions: List[str], doc_id: str) -> List[Dict]:
        """Process multiple questions efficiently"""
        try:
            start_time = time.time()
            logger.info(f"📋 Processing {len(questions)} questions in batch")
            
            # Process questions concurrently (limited concurrency to avoid rate limits)
            semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent queries
            
            async def process_single_query(question: str, index: int) -> Dict:
                async with semaphore:
                    answer = await self.query(question, doc_id)
                    return {
                        "question_index": index + 1,
                        "question": question,
                        "answer": answer
                    }
            
            tasks = [process_single_query(q, i) for i, q in enumerate(questions)]
            results = await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
            logger.info(f"✅ Batch processing completed in {total_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch query error: {e}")
            return [{"error": str(e)}]

    def get_document_stats(self, doc_id: str) -> Dict:
        """Get statistics about a processed document"""
        try:
            namespace = self.create_namespace(doc_id)
            stats = self.index.describe_index_stats(filter={'doc_id': doc_id})
            
            return {
                "doc_id": doc_id,
                "namespace": namespace,
                "vector_count": stats.get('total_vector_count', 0),
                "status": "processed" if stats.get('total_vector_count', 0) > 0 else "not_found"
            }
            
        except Exception as e:
            logger.error(f"❌ Stats error: {e}")
            return {"error": str(e)}

    def delete_document(self, doc_id: str) -> bool:
        """Delete all vectors for a specific document"""
        try:
            namespace = self.create_namespace(doc_id)
            
            # Delete by namespace (if supported) or by filter
            self.index.delete(delete_all=True, namespace=namespace)
            
            logger.info(f"🗑️ Deleted document {doc_id} from namespace {namespace}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Document deletion failed: {e}")
            return False

