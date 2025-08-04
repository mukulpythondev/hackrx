import os
import asyncio
import hashlib
import tempfile
import aiohttp
from typing import Dict, List
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import concurrent.futures

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from app.prompt import build_chat_prompt
from .utils import format_docs

# Load environment variables
load_dotenv()

# Global cache for processed documents and embeddings
PROCESSED_DOCS: Dict[str, bool] = {}
EMBEDDING_CACHE: Dict[str, List[float]] = {}

# Pinecone setup
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
NAMESPACE = "hackrx"

if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# OPTIMIZED: Better model configurations for accuracy
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",  # Better accuracy than small
    chunk_size=500,  # Reduced for faster processing
    max_retries=3,
    request_timeout=30
)

llm = ChatOpenAI(
    model="gpt-4o",  # Much better than gpt-4o-mini for this task
    temperature=0,
    max_tokens=500,  # Increased for complete answers
    timeout=30, 
    max_retries=3
)


def generate_document_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:16]

async def check_document_exists_async(doc_id: str) -> bool:
    """Optimized document existence check using fetch instead of query"""
    try:
        loop = asyncio.get_event_loop()
        
        # Use fetch with a specific vector ID (much faster than query)
        test_vector_id = f"{doc_id}_chunk_0"  # Standard first chunk ID
        
        result = await loop.run_in_executor(
            None,
            lambda: index.fetch(
                ids=[test_vector_id],
                namespace=NAMESPACE
            )
        )
        
        exists = len(result.get("vectors", {})) > 0
        print(f"🔍 Document {doc_id} exists: {exists}")
        return exists
        
    except Exception as e:
        print(f"[!] Error checking document existence with fetch: {e}")
        # Fallback to query method if fetch fails
        try:
            result = await loop.run_in_executor(
                None,
                lambda: index.query(
                    vector=[0.1] * 3072,
                    filter={"document_id": doc_id},
                    top_k=1,
                    namespace=NAMESPACE,
                    include_metadata=False  # Don't need metadata for existence check
                )
            )
            exists = len(result.get("matches", [])) > 0
            print(f"🔍 Document {doc_id} exists (fallback): {exists}")
            return exists
        except Exception as fallback_e:
            print(f"[!] Fallback query also failed: {fallback_e}")
            return False

async def download_pdf_async(url: str) -> str:
    """Async PDF download with timeout"""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                content = await resp.read()
                
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            return tmp.name
    except Exception as e:
        raise Exception(f"Download error: {e}")

def process_document_chunks(temp_file: str) -> List[Document]:
    """Optimized document processing with smart chunking"""
    loader = PyPDFLoader(temp_file)
    docs = loader.load()
    
    # Better chunking strategy for insurance documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Optimal for insurance policy Q&A
        chunk_overlap=300,  # Good context preservation
        separators=[
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            ", ",    # Clauses
            " ",     # Words
            ""       # Characters
        ],
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = splitter.split_documents(docs)
    
    # Filter out very short or empty chunks
    filtered_chunks = []
    for chunk in chunks:
        content = chunk.page_content.strip()
        if len(content) >= 50 and not content.isspace():  # Minimum meaningful content
            filtered_chunks.append(chunk)
    
    print(f"📄 Created {len(filtered_chunks)} meaningful chunks from {len(chunks)} total")
    return filtered_chunks

async def batch_embed_queries(questions: List[str]) -> Dict[str, List[float]]:
    """Batch embed all questions at once for efficiency"""
    cached_embeddings = {}
    uncached_questions = []
    
    # Check cache first
    for q in questions:
        if q in EMBEDDING_CACHE:
            cached_embeddings[q] = EMBEDDING_CACHE[q]
        else:
            uncached_questions.append(q)
    
    # Batch embed uncached questions
    if uncached_questions:
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            embedding_results = await loop.run_in_executor(
                executor, embeddings.embed_documents, uncached_questions
            )
        
        # Update cache and results
        for q, emb in zip(uncached_questions, embedding_results):
            EMBEDDING_CACHE[q] = emb
            cached_embeddings[q] = emb
    
    return cached_embeddings

async def enhanced_retrieve(question: str, doc_id: str, k: int = 6) -> List[Document]:
    """Enhanced retrieval with score filtering and reranking"""
    try:
        # Get cached embedding or compute it
        question_embeddings = await batch_embed_queries([question])
        question_embedding = question_embeddings[question]
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: index.query(
                vector=question_embedding,
                top_k=k * 2,  # Get more initially for filtering
                namespace=NAMESPACE,
                include_metadata=True,
                filter={"document_id": doc_id}
            )
        )
        
        # Filter by relevance score (adjust threshold as needed)
        filtered_matches = [
            match for match in result.matches 
            if match.score > 0.5  # Only high-confidence matches
        ]
        
        # If too few high-confidence matches, lower threshold
        if len(filtered_matches) < 3:
            filtered_matches = [
                match for match in result.matches 
                if match.score > 0.65
            ]
        
        # Convert to documents and sort by score
        docs = []
        for match in filtered_matches[:k]:  # Limit to k results
            if "text" in match.metadata:
                doc = Document(
                    page_content=match.metadata["text"],
                    metadata={
                        **match.metadata,
                        "score": match.score
                    }
                )
                docs.append(doc)
        
        print(f"🔍 Found {len(docs)} high-quality chunks for question: {question[:50]}... (scores: {[f'{d.metadata.get('score', 0):.3f}' for d in docs[:3]]})")
        return docs
        
    except Exception as e:
        print(f"[!] Enhanced retrieval failed for '{question[:30]}...': {e}")
        return []

def build_optimized_rag_chain(splits: List[Document], doc_id: str):
    """Optimized RAG chain with consistent vector IDs and proper caching"""
    vector_store = PineconeVectorStore(
        index=index, 
        embedding=embeddings, 
        namespace=NAMESPACE
    )

    # FIXED: Only upload if we actually have new splits to process
    if splits and len(splits) > 0:
        print(f"📤 Uploading {len(splits)} new chunks to vector database...")
        
        # Prepare documents with consistent IDs and metadata
        docs_to_upload = []
        vector_ids = []
        
        for i, split in enumerate(splits):
            # Create consistent vector ID
            vector_id = f"{doc_id}_chunk_{i}"
            vector_ids.append(vector_id)
            
            split.metadata.update({
                "document_id": doc_id,
                "chunk_index": i,
                "vector_id": vector_id,
                "text": split.page_content[:2000],  # Truncate if too long for metadata
                "chunk_size": len(split.page_content)
            })
            docs_to_upload.append(split)
        
        # Upload with explicit IDs for consistent caching
        vector_store.add_documents(
            documents=docs_to_upload,
            ids=vector_ids
        )
        
        PROCESSED_DOCS[doc_id] = True
        print(f"✅ Uploaded {len(docs_to_upload)} chunks with consistent IDs")
    elif doc_id in PROCESSED_DOCS:
        print(f"⚡ Using existing document chunks from vector database")
    else:
        print(f"⚠️ No splits provided and document not in cache - this shouldn't happen")

    async def batch_retrieve(questions: List[str]) -> Dict[str, List[Document]]:
        """Retrieve documents for all questions using enhanced retrieval"""        
        results = {}
        
        # Execute all retrievals concurrently
        tasks = [enhanced_retrieve(q, doc_id) for q in questions]
        retrieval_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for q, docs in zip(questions, retrieval_results):
            if isinstance(docs, Exception):
                print(f"[!] Retrieval exception for '{q[:30]}...': {docs}")
                results[q] = []
            else:
                results[q] = docs
        
        return results

    prompt = build_chat_prompt()

    async def batch_rag_pipeline(questions: List[str]) -> List[str]:
        """Process all questions in optimized batches"""
        # Batch retrieve all contexts
        question_docs = await batch_retrieve(questions)
        
        # Prepare all prompts
        formatted_prompts = []
        for question in questions:
            docs = question_docs.get(question, [])
            context = format_docs(docs)
            formatted_prompt = prompt.format_messages(context=context, question=question)
            formatted_prompts.append(formatted_prompt)
        
        # Batch process with LLM (concurrent but limited)
        async def process_single_llm(prompt_msgs, question_idx, question):
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, lambda: llm.invoke(prompt_msgs)
                )
                result = response.content.strip()
                
                # Clean up the answer
                if result and not result.endswith(('.', '!', '?')):
                    result += "."
                
                print(f"✅ Q{question_idx + 1} answered: {result[:100]}...")
                return result
            except Exception as e:
                print(f"[!] LLM processing failed for Q{question_idx + 1}: {e}")
                return "Error: Unable to process this question."
        
        # Process in smaller concurrent batches to avoid rate limits
        batch_size = 3  # Conservative batch size for stability
        answers = []
        
        for i in range(0, len(formatted_prompts), batch_size):
            batch = formatted_prompts[i:i + batch_size]
            batch_questions = questions[i:i + batch_size]
            batch_indices = list(range(i, min(i + batch_size, len(formatted_prompts))))
            
            batch_tasks = [
                process_single_llm(prompt, idx, q) 
                for prompt, idx, q in zip(batch, batch_indices, batch_questions)
            ]
            batch_answers = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for answer in batch_answers:
                if isinstance(answer, Exception):
                    answers.append("Error: Processing failed.")
                else:
                    answers.append(answer)
        
        return answers

    return batch_rag_pipeline

async def process_query(pdf_url: str, questions: List[str]) -> List[str]:
    """Optimized main processing function with proper caching"""
    doc_id = generate_document_id(pdf_url)
    temp_file = None

    try:
        print(f"📋 Document ID: {doc_id}")
        
        # FIXED: Proper caching logic
        needs_processing = False
        
        # Check in-memory cache first
        if doc_id in PROCESSED_DOCS:
            print(f"✅ Document {doc_id} found in memory cache")
            needs_processing = False
        else:
            # Check if document exists in Pinecone
            print(f"🔍 Checking if document exists in vector database...")
            doc_exists = await check_document_exists_async(doc_id)
            
            if doc_exists:
                print(f"✅ Document {doc_id} found in vector database, skipping download")
                PROCESSED_DOCS[doc_id] = True  # Update memory cache
                needs_processing = False
            else:
                print(f"📥 Document {doc_id} not found, needs processing")
                needs_processing = True

        splits = []
        if needs_processing:
            print(f"📥 Downloading and processing document...")
            # Download PDF asynchronously
            temp_file = await download_pdf_async(pdf_url)
            
            # Process document in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                splits = await loop.run_in_executor(executor, process_document_chunks, temp_file)
            
            print(f"📄 Created {len(splits)} document chunks")
        else:
            print(f"⚡ Using cached document, skipping download and processing")

        # Build optimized RAG chain
        rag_chain = build_optimized_rag_chain(splits, doc_id)
        
        print(f"🔍 Processing {len(questions)} questions...")
        # Process all questions in optimized batches  
        answers = await rag_chain(questions)
        
        print(f"✅ Completed processing {len(answers)} answers")
        return answers

    except Exception as e:
        print(f"[!] process_query error: {e}")
        return [f"Error: {e}"] * len(questions)

    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as e:
                print(f"[!] Cleanup error: {e}")