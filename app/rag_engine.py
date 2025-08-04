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
# Load environment variables
load_dotenv()

# Global cache for processed documents and embeddings
PROCESSED_DOCS: Dict[str, bool] = {}
EMBEDDING_CACHE: Dict[str, List[float]] = {}

# Pinecone setup
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# Model configurations
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    chunk_size=500,
    max_retries=3,
    request_timeout=30
)

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=800,
    timeout=30, 
    max_retries=3
)

def generate_document_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:16]

def get_document_namespace(doc_id: str) -> str:
    """CRITICAL FIX: Use document-specific namespace to prevent contamination"""
    return f"doc_{doc_id}"

async def check_document_exists_async(doc_id: str) -> bool:
    """Check if document exists using document-specific namespace"""
    try:
        doc_namespace = get_document_namespace(doc_id)
        loop = asyncio.get_event_loop()
        
        # Try to query the document-specific namespace
        result = await loop.run_in_executor(
            None,
            lambda: index.query(
                vector=[0.1] * 3072,
                top_k=1,
                namespace=doc_namespace,  # Document-specific namespace
                include_metadata=False
            )
        )
        
        exists = len(result.matches) > 0
        print(f"🔍 Document {doc_id} exists in namespace {doc_namespace}: {exists}")
        return exists
        
    except Exception as e:
        print(f"[!] Error checking document existence: {e}")
        return False

async def download_pdf_async(url: str) -> str:
    """Async PDF download with timeout"""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                content = await resp.read()
                
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            return tmp.name
    except Exception as e:
        raise Exception(f"Download error: {e}")

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    import re
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Basic cleaning
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\%\$]', ' ', text)
    return text.strip()

def process_document_chunks(temp_file: str) -> List[Document]:
    """Process document into chunks with optimal settings"""
    loader = PyPDFLoader(temp_file)
    docs = loader.load()
    
    # Combine all pages
    full_text = "\n\n".join([doc.page_content for doc in docs])
    full_text = clean_text(full_text)
    
    # Optimal chunking for insurance documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Good balance of context and specificity
        chunk_overlap=200,  # Sufficient overlap
        separators=[
            "\n\n\n",   # Major sections
            "\n\n",     # Paragraphs
            "\n",       # Lines
            ". ",       # Sentences
            "; ",       # Semi-colons
            ", ",       # Clauses
            " ",        # Words
            ""          # Characters
        ],
        length_function=len,
        is_separator_regex=False,
    )
    
    # Create document object
    full_doc = Document(page_content=full_text, metadata={"source": temp_file})
    chunks = splitter.split_documents([full_doc])
    
    # Filter meaningful chunks
    filtered_chunks = []
    for i, chunk in enumerate(chunks):
        content = chunk.page_content.strip()
        
        # Skip very short chunks
        if len(content) < 100:
            continue
            
        # Add metadata
        chunk.metadata.update({
            "chunk_id": i,
            "chunk_length": len(content),
            "word_count": len(content.split())
        })
        
        filtered_chunks.append(chunk)
    
    print(f"📄 Created {len(filtered_chunks)} chunks from {len(chunks)} total")
    return filtered_chunks

async def batch_embed_queries(questions: List[str]) -> Dict[str, List[float]]:
    """Batch embed questions with caching"""
    cached_embeddings = {}
    uncached_questions = []
    
    for q in questions:
        if q in EMBEDDING_CACHE:
            cached_embeddings[q] = EMBEDDING_CACHE[q]
        else:
            uncached_questions.append(q)
    
    if uncached_questions:
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            embedding_results = await loop.run_in_executor(
                executor, embeddings.embed_documents, uncached_questions
            )
        
        for q, emb in zip(uncached_questions, embedding_results):
            EMBEDDING_CACHE[q] = emb
            cached_embeddings[q] = emb
    
    return cached_embeddings

async def enhanced_retrieve(question: str, doc_id: str, k: int = 8) -> List[Document]:
    """FIXED: Retrieve from document-specific namespace only"""
    try:
        doc_namespace = get_document_namespace(doc_id)
        
        # Get question embedding
        question_embeddings = await batch_embed_queries([question])
        question_embedding = question_embeddings[question]
        
        loop = asyncio.get_event_loop()
        
        print(f"🔍 Searching in namespace: {doc_namespace} for question: {question[:50]}...")
        
        result = await loop.run_in_executor(
            None,
            lambda: index.query(
                vector=question_embedding,
                top_k=k * 2,  # Get more for filtering
                namespace=doc_namespace,  # CRITICAL: Document-specific namespace
                include_metadata=True
                # No document_id filter needed - namespace isolates documents
            )
        )
        
        print(f"📊 Found {len(result.matches)} matches in namespace {doc_namespace}")
        
        if result.matches:
            print(f"📊 Scores: {[f'{m.score:.3f}' for m in result.matches[:5]]}")
        
        # Apply reasonable score threshold
        min_score = 0.3  # Lenient threshold
        filtered_matches = [m for m in result.matches if m.score > min_score]
        
        # If too few matches, take top matches regardless
        if len(filtered_matches) < 3 and result.matches:
            filtered_matches = result.matches[:k]
            print(f"🔄 Using top {len(filtered_matches)} matches regardless of score")
        
        # Convert to documents
        docs = []
        for match in filtered_matches[:k]:
            if match.metadata and "text" in match.metadata:
                doc = Document(
                    page_content=match.metadata["text"],
                    metadata={
                        **match.metadata,
                        "score": match.score
                    }
                )
                docs.append(doc)
        
        print(f"✅ Retrieved {len(docs)} documents for question")
        return docs
        
    except Exception as e:
        print(f"[!] Retrieval failed for '{question[:30]}...': {e}")
        import traceback
        traceback.print_exc()
        return []

def build_optimized_rag_chain(splits: List[Document], doc_id: str):
    """FIXED: Use document-specific namespace"""
    doc_namespace = get_document_namespace(doc_id)
    
    vector_store = PineconeVectorStore(
        index=index, 
        embedding=embeddings, 
        namespace=doc_namespace  # CRITICAL: Document-specific namespace
    )

    if splits and len(splits) > 0:
        print(f"📤 Uploading {len(splits)} chunks to namespace: {doc_namespace}")
        
        docs_to_upload = []
        vector_ids = []
        
        for i, split in enumerate(splits):
            vector_id = f"chunk_{i}"  # Simpler ID since namespace isolates documents
            vector_ids.append(vector_id)
            
            # Store full text in metadata
            text_content = split.page_content
            if len(text_content) > 40000:  # Pinecone metadata limit
                text_content = text_content[:40000]
            
            split.metadata.update({
                "chunk_index": i,
                "vector_id": vector_id,
                "text": text_content,
                "chunk_size": len(split.page_content)
            })
            docs_to_upload.append(split)
        
        try:
            vector_store.add_documents(
                documents=docs_to_upload,
                ids=vector_ids
            )
            print(f"✅ Successfully uploaded to namespace {doc_namespace}")
            
            # Verification
            test_result = index.query(
                vector=[0.1] * 3072,
                top_k=3,
                namespace=doc_namespace,
                include_metadata=True
            )
            print(f"🧪 Verification: {len(test_result.matches)} chunks in namespace")
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            import traceback
            traceback.print_exc()
        
        PROCESSED_DOCS[doc_id] = True

    async def batch_retrieve(questions: List[str]) -> Dict[str, List[Document]]:
        """Retrieve from document-specific namespace"""        
        results = {}
        
        for q in questions:
            docs = await enhanced_retrieve(q, doc_id, k=8)
            results[q] = docs
            
            if not docs:
                print(f"⚠️ No documents retrieved for: '{q[:50]}...'")
        
        return results

    async def batch_rag_pipeline(questions: List[str]) -> List[str]:
        """Process questions with fixed prompt"""
        question_docs = await batch_retrieve(questions)
        
        answers = []
        prompt = build_chat_prompt()
        
        for i, question in enumerate(questions):
            try:
                docs = question_docs.get(question, [])
                
                if docs:
                    # Format context from retrieved docs
                    context_parts = []
                    for j, doc in enumerate(docs[:6]):
                        score = doc.metadata.get('score', 0)
                        content = doc.page_content[:1500]
                        context_parts.append(f"[Relevant Section {j+1}]:\n{content}\n")
                    
                    context = "\n".join(context_parts)
                else:
                    context = "No relevant information found in the document."
                
                # Use the fixed prompt
                formatted_prompt = prompt.format_messages(context=context, question=question)
                
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: llm.invoke(formatted_prompt)
                )
                
                result = response.content.strip()
                
                if result and not result.endswith(('.', '!', '?')):
                    result += "."
                
                answers.append(result)
                print(f"✅ Q{i + 1} answered: {result[:100]}...")
                
            except Exception as e:
                print(f"[!] Error processing Q{i + 1}: {e}")
                answers.append("Error: Unable to process this question.")
                
            await asyncio.sleep(0.1)
        
        return answers

    return batch_rag_pipeline

async def process_query(pdf_url: str, questions: List[str]) -> List[str]:
    """FIXED: Main processing with document-specific namespace"""
    doc_id = generate_document_id(pdf_url)
    temp_file = None

    try:
        print(f"📋 Processing Document ID: {doc_id}")
        print(f"📋 Using namespace: {get_document_namespace(doc_id)}")
        
        # Check if document exists in its specific namespace
        needs_processing = False
        
        if doc_id in PROCESSED_DOCS:
            print(f"✅ Document {doc_id} found in memory cache")
            needs_processing = False
        else:
            doc_exists = await check_document_exists_async(doc_id)
            
            if doc_exists:
                print(f"✅ Document {doc_id} found in vector database")
                PROCESSED_DOCS[doc_id] = True
                needs_processing = False
            else:
                print(f"📥 Document {doc_id} needs processing")
                needs_processing = True

        splits = []
        if needs_processing:
            print(f"📥 Downloading and processing document...")
            temp_file = await download_pdf_async(pdf_url)
            
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                splits = await loop.run_in_executor(executor, process_document_chunks, temp_file)

        # Build RAG chain with document-specific namespace
        rag_chain = build_optimized_rag_chain(splits, doc_id)
        
        print(f"🔍 Processing {len(questions)} questions...")
        answers = await rag_chain(questions)
        
        print(f"✅ Completed processing {len(answers)} answers")
        return answers

    except Exception as e:
        print(f"[!] process_query error: {e}")
        return [f"Error processing question: {e}"] * len(questions)

    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as e:
                print(f"[!] Cleanup error: {e}")