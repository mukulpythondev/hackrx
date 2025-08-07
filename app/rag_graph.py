import os
import asyncio
import hashlib
import tempfile
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import logging
import time

# Core imports
from neo4j import GraphDatabase
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

class HybridRAGSystem:
    def __init__(self):
        try:
            # Initialize Neo4j
            self.neo4j_driver = GraphDatabase.driver(
                os.environ["NEO4J_URL"],
                auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
            )
            logger.info("✅ Neo4j connected")

            # Initialize Pinecone
            self.pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
            index_name = os.environ["PINECONE_INDEX_NAME"]
            dimension = 3072  # text-embedding-3-large dimension
            cloud = "aws"
            region = "us-east-1"

            # Check if index exists, else create
            existing_indexes = [i["name"] for i in self.pinecone.list_indexes()]
            if index_name not in existing_indexes:
                self.pinecone.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=cloud, region=region)
                )
                logger.info(f"✅ Created Pinecone index '{index_name}'")
                # Wait for index to be ready
                time.sleep(10)

            self.index = self.pinecone.Index(index_name)

            # Initialize embeddings and LLM - FIXED: Ensure correct embedding model
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

            logger.info("✅ Hybrid RAG system initialized")

        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise

    async def download_pdf(self, url: str) -> str:
        """Download PDF to temp file"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to download PDF: HTTP {resp.status}")
                    content = await resp.read()
                    
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(content)
                logger.info(f"📥 Downloaded PDF to {tmp.name}")
                return tmp.name
                
        except Exception as e:
            logger.error(f"❌ PDF download failed: {e}")
            raise

    def extract_and_chunk_pdf(self, temp_file: str, doc_id: str) -> List[Document]:
        """Extract text and create chunks"""
        try:
            loader = PyPDFLoader(temp_file)
            docs = loader.load()
            
            if not docs:
                raise Exception("No content extracted from PDF")
            
            # Simple text splitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Increased for better context
                chunk_overlap=100
            )
            
            chunks = splitter.split_documents(docs)
            
            # Add metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'doc_id': doc_id,
                    'chunk_id': f"{doc_id}_{i}",
                    'chunk_index': i,
                    'page': chunk.metadata.get('page', 0)
                })
            
            logger.info(f"📄 Extracted {len(chunks)} chunks from PDF")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ PDF extraction failed: {e}")
            raise

    def create_knowledge_graph(self, chunks: List[Document], doc_id: str):
        """Create knowledge graph from document chunks"""
        try:
            with self.neo4j_driver.session() as session:
                # Create document node
                session.run(
                    "MERGE (d:Document {id: $doc_id}) SET d.title = $title",
                    doc_id=doc_id, 
                    title=f"Document_{doc_id}"
                )
                
                # Create chunk nodes and relationships
                for chunk in chunks:
                    chunk_id = chunk.metadata['chunk_id']
                    text = chunk.page_content[:1500]  # Increased limit
                    
                    # Create chunk node
                    session.run("""
                        MERGE (c:Chunk {id: $chunk_id})
                        SET c.doc_id = $doc_id,
                            c.text = $text,
                            c.chunk_index = $chunk_index,
                            c.page = $page
                    """, 
                    chunk_id=chunk_id, 
                    doc_id=doc_id,
                    text=text,
                    chunk_index=chunk.metadata['chunk_index'],
                    page=chunk.metadata.get('page', 0)
                    )
                    
                    # Connect chunk to document
                    session.run("""
                        MATCH (d:Document {id: $doc_id})
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (d)-[:CONTAINS]->(c)
                    """, doc_id=doc_id, chunk_id=chunk_id)
                    
                    # Connect adjacent chunks
                    if chunk.metadata['chunk_index'] > 0:
                        prev_chunk_id = f"{doc_id}_{chunk.metadata['chunk_index'] - 1}"
                        session.run("""
                            MATCH (c1:Chunk {id: $prev_chunk_id})
                            MATCH (c2:Chunk {id: $chunk_id})
                            MERGE (c1)-[:NEXT]->(c2)
                        """, prev_chunk_id=prev_chunk_id, chunk_id=chunk_id)
            
            logger.info(f"📊 Created knowledge graph with {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"❌ Graph creation failed: {e}")
            raise

    async def store_vectors(self, chunks: List[Document], batch_size: int = 50):
        """Store chunk embeddings into Pinecone in batches"""
        try:
            # Generate embeddings
            texts = [chunk.page_content for chunk in chunks]
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None, self.embeddings.embed_documents, texts
            )
            
            # Prepare vectors
            vectors_to_upsert = []
            for chunk, embedding in zip(chunks, embeddings):
                vectors_to_upsert.append({
                    "id": chunk.metadata["chunk_id"],
                    "values": embedding,
                    "metadata": {
                        "text": chunk.page_content[:1000],  # Pinecone metadata limit
                        "doc_id": chunk.metadata["doc_id"],
                        "chunk_index": chunk.metadata["chunk_index"],
                        "page": chunk.metadata.get("page", 0)
                    }
                })

            # Batch upserts
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                try:
                    self.index.upsert(vectors=batch)
                    logger.info(f"📤 Upserted batch {i // batch_size + 1} with {len(batch)} vectors")
                except Exception as e:
                    logger.error(f"🛑 Pinecone upsert failed on batch {i // batch_size + 1}: {e}")
                    time.sleep(1)
                    
            logger.info(f"✅ Stored {len(vectors_to_upsert)} vectors in Pinecone")
            
        except Exception as e:
            logger.error(f"❌ Vector storage failed: {e}")
            raise

    async def vector_search(self, query: str, doc_id: str, top_k: int = 5) -> List[Dict]:
        """Search using vector similarity"""
        try:
            query_embedding = await asyncio.get_event_loop().run_in_executor(
                None, self.embeddings.embed_query, query
            )
            
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter={'doc_id': doc_id},
                include_metadata=True
            )
            
            return [
                {
                    'chunk_id': match.id,
                    'text': match.metadata['text'],
                    'score': match.score,
                    'chunk_index': match.metadata.get('chunk_index', 0)
                }
                for match in results.matches
            ]
            
        except Exception as e:
            logger.error(f"❌ Vector search failed: {e}")
            return []

    def graph_search(self, search_query: str, doc_id: str) -> List[Dict]:
        """Search using graph traversal and text matching - FIXED parameter naming"""
        try:
            with self.neo4j_driver.session() as session:
                # FIXED: Renamed parameter from 'query' to 'search_query' to avoid conflict
                result = session.run("""
                    MATCH (d:Document {id: $doc_id})-[:CONTAINS]->(c:Chunk)
                    WHERE toLower(c.text) CONTAINS toLower($search_query)
                    RETURN c.id as chunk_id, c.text as text, c.chunk_index as chunk_index
                    ORDER BY c.chunk_index
                    LIMIT 5
                """, doc_id=doc_id, search_query=search_query)
                
                chunks = []
                for record in result:
                    chunks.append({
                        'chunk_id': record['chunk_id'],
                        'text': record['text'],
                        'chunk_index': record['chunk_index']
                    })
                
                # Get enhanced context for each chunk
                enhanced_chunks = []
                for chunk in chunks:
                    context_result = session.run("""
                        MATCH (c:Chunk {id: $chunk_id})
                        OPTIONAL MATCH (prev:Chunk)-[:NEXT]->(c)
                        OPTIONAL MATCH (c)-[:NEXT]->(next:Chunk)
                        RETURN c.text as current, prev.text as previous, next.text as next_text
                    """, chunk_id=chunk['chunk_id'])
                    
                    record = context_result.single()
                    if record:
                        context = ""
                        if record['previous']:
                            context += record['previous'] + " "
                        context += record['current']
                        if record['next_text']:
                            context += " " + record['next_text']
                        
                        enhanced_chunks.append({
                            'chunk_id': chunk['chunk_id'],
                            'text': context,
                            'chunk_index': chunk['chunk_index']
                        })
                
                logger.info(f"📊 Graph search found {len(enhanced_chunks)} chunks")
                return enhanced_chunks
                
        except Exception as e:
            logger.error(f"❌ Graph search failed: {e}")
            return []

    async def hybrid_search(self, query: str, doc_id: str) -> List[Document]:
        """Combine vector and graph search results"""
        try:
            # Run both searches concurrently for speed
            vector_task = self.vector_search(query, doc_id, top_k=4)
            graph_task = asyncio.get_event_loop().run_in_executor(
                None, self.graph_search, query, doc_id
            )
            
            vector_results, graph_results = await asyncio.gather(vector_task, graph_task)
            
            # Combine and deduplicate
            seen_chunks = set()
            combined_results = []
            
            # Prioritize vector search results
            for result in vector_results:
                if result['chunk_id'] not in seen_chunks:
                    combined_results.append(Document(
                        page_content=result['text'],
                        metadata={
                            'chunk_id': result['chunk_id'],
                            'source': 'vector',
                            'score': result['score'],
                            'chunk_index': result.get('chunk_index', 0)
                        }
                    ))
                    seen_chunks.add(result['chunk_id'])
            
            # Add graph results
            for result in graph_results:
                if result['chunk_id'] not in seen_chunks:
                    combined_results.append(Document(
                        page_content=result['text'],
                        metadata={
                            'chunk_id': result['chunk_id'],
                            'source': 'graph',
                            'chunk_index': result['chunk_index']
                        }
                    ))
                    seen_chunks.add(result['chunk_id'])
            
            logger.info(f"🔍 Hybrid search: {len(vector_results)} vector + {len(graph_results)} graph = {len(combined_results)} total")
            return combined_results[:5]  # Return top 5
            
        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {e}")
            return []

    async def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        """Generate answer using LLM"""
        try:
            if not context_docs:
                return "No relevant information found in the document."
            
            context = "\n\n".join([
                f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}" 
                for doc in context_docs
            ])
            
            # Enhanced prompt for better answers
            prompt = f"""You are an expert assistant analyzing a policy document. Based on the following context from the document, provide a comprehensive and accurate answer to the question. 

If the information is available in the context:
- Provide specific details with exact numbers, percentages, or timeframes when mentioned
- Quote relevant policy terms or conditions
- Structure your answer clearly

If the information is not available in the context, clearly state that the information is not found in the provided sections.

Context:
{context}

Question: {query}

Answer:"""
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.llm.invoke, prompt
            )
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"❌ Answer generation failed: {e}")
            return f"Error generating answer: {str(e)}"

    async def process_document(self, pdf_url: str, doc_id: Optional[str] = None) -> str:
        """Process PDF and store in both vector and graph databases"""
        if not doc_id:
            doc_id = hashlib.md5(pdf_url.encode()).hexdigest()[:12]
        
        temp_file = None
        try:
            logger.info(f"🚀 Processing document: {pdf_url}")
            
            # Download and process PDF
            temp_file = await self.download_pdf(pdf_url)
            chunks = self.extract_and_chunk_pdf(temp_file, doc_id)
            
            if not chunks:
                raise Exception("No chunks extracted from PDF")
            
            # Store in both databases concurrently for speed
            graph_task = asyncio.get_event_loop().run_in_executor(
                None, self.create_knowledge_graph, chunks, doc_id
            )
            vector_task = self.store_vectors(chunks)
            
            await asyncio.gather(graph_task, vector_task)
            
            logger.info(f"✅ Document processed successfully: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            raise
            
        finally:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)

    def check_document_exists(self, doc_id: str) -> bool:
        """Check if document already exists in the system"""
        try:
            # Check in Pinecone
            stats = self.index.describe_index_stats()
            if stats.total_vector_count > 0:
                # Try a small query to see if vectors exist for this doc_id
                dummy_vector = [0.0] * 3072  # Create a dummy vector for testing
                results = self.index.query(
                    vector=dummy_vector,
                    top_k=1,
                    filter={'doc_id': doc_id},
                    include_metadata=True
                )
                return len(results.matches) > 0
            return False
        except Exception as e:
            logger.warning(f"⚠️ Error checking document existence: {e}")
            return False

    async def query(self, question: str, doc_id: str) -> str:
        """Main query interface using hybrid search"""
        try:
            start_time = time.time()
            logger.info(f"❓ Processing query: {question}")
            
            # Hybrid search
            context_docs = await self.hybrid_search(question, doc_id)
            
            # Generate answer
            answer = await self.generate_answer(question, context_docs)
            
            elapsed_time = time.time() - start_time
            logger.info(f"⏱️ Query processed in {elapsed_time:.2f} seconds")
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ Query error: {e}")
            return f"Error processing query: {str(e)}"

    def close(self):
        """Clean up resources"""
        try:
            self.neo4j_driver.close()
            logger.info("✅ Resources cleaned up")
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

# Example usage for testing
async def main():
    rag = HybridRAGSystem()
    
    try:
        # Example PDF URL (replace with your actual URL)
        pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
        
        # Process document
        doc_id = await rag.process_document(pdf_url)
        print(f"Document processed with ID: {doc_id}")
        
        # Example queries
        questions = [
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?",
            "Does this policy cover maternity expenses?"
        ]
        
        for question in questions:
            answer = await rag.query(question, doc_id)
            print(f"\nQ: {question}")
            print(f"A: {answer}")
            
    finally:
        rag.close()

if __name__ == "__main__":
    asyncio.run(main())