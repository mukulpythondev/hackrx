import os
import gc
import asyncio
import hashlib
import tempfile
import aiohttp
import logging
import time
import re
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import concurrent.futures
from contextlib import asynccontextmanager
import numpy as np
from dataclasses import dataclass

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from app.prompt import build_chat_prompt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# MEMORY MANAGEMENT: Limited caches with size controls
MAX_CACHE_SIZE = 50
MAX_EMBEDDING_CACHE_SIZE = 100

@dataclass
class ChunkMetadata:
    """Enhanced metadata for document chunks"""
    chunk_id: str
    section_title: str
    chunk_type: str  # 'header', 'paragraph', 'list', 'table'
    importance_score: float
    position: int
    word_count: int

class LimitedDict(dict):
    """Dictionary with size limit to prevent memory leaks"""
    def __init__(self, max_size: int):
        super().__init__()
        self.max_size = max_size
    
    def __setitem__(self, key, value):
        if len(self) >= self.max_size:
            # Remove oldest item (FIFO)
            oldest = next(iter(self))
            del self[oldest]
            logger.info(f"🧹 Cache cleanup: removed {oldest}")
        super().__setitem__(key, value)

# Global cache with memory limits
PROCESSED_DOCS: LimitedDict = LimitedDict(MAX_CACHE_SIZE)
EMBEDDING_CACHE: LimitedDict = LimitedDict(MAX_EMBEDDING_CACHE_SIZE)

class EnhancedTextProcessor:
    """Advanced text processing for better chunking"""
    
    @staticmethod
    def detect_document_structure(text: str) -> Dict[str, List[str]]:
        """Detect document structure (headers, sections, etc.)"""
        structure = {
            'headers': [],
            'paragraphs': [],
            'lists': [],
            'tables': []
        }
        
        lines = text.split('\n')
        current_section = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect headers (various patterns)
            if (re.match(r'^[A-Z\s]{3,}$', line) or  # ALL CAPS
                re.match(r'^\d+\.\s+[A-Z]', line) or  # Numbered headers
                re.match(r'^[IVX]+\.\s+', line) or    # Roman numerals
                len(line.split()) <= 6 and line.endswith(':')):  # Short lines ending with :
                structure['headers'].append(line)
                current_section = line
                
            # Detect lists
            elif re.match(r'^[\-\*\+•]\s+', line) or re.match(r'^\d+[\.\)]\s+', line):
                structure['lists'].append((current_section, line))
                
            # Detect potential tables (simple heuristic)
            elif '|' in line and line.count('|') >= 2:
                structure['tables'].append(line)
                
            # Regular paragraphs
            else:
                structure['paragraphs'].append((current_section, line))
        
        return structure
    
    @staticmethod
    def calculate_importance_score(text: str, doc_structure: Dict) -> float:
        """Calculate importance score for a chunk"""
        score = 1.0
        
        # Boost score for chunks with important keywords (legal/HR/insurance)
        important_keywords = [
            'policy', 'coverage', 'premium', 'claim', 'benefit', 'liability',
            'contract', 'agreement', 'terms', 'conditions', 'legal', 'compliance',
            'employee', 'employer', 'salary', 'leave', 'vacation', 'harassment'
        ]
        
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in important_keywords if keyword in text_lower)
        score += keyword_count * 0.1
        
        # Boost score for structured content
        if any(header in text for header in doc_structure.get('headers', [])):
            score += 0.3
            
        # Boost score for longer, more informative chunks
        if len(text.split()) > 50:
            score += 0.2
            
        return min(score, 2.0)  # Cap at 2.0

class QueryExpander:
    """Enhanced query expansion for better retrieval"""
    
    @staticmethod
    def expand_legal_query(query: str) -> List[str]:
        """Expand queries with legal synonyms and related terms"""
        legal_expansions = {
            'contract': ['agreement', 'terms', 'conditions', 'covenant'],
            'liability': ['responsibility', 'obligation', 'duty', 'accountability'],
            'compliance': ['adherence', 'conformity', 'observance', 'regulation'],
            'policy': ['procedure', 'guideline', 'rule', 'regulation'],
            'claim': ['request', 'demand', 'assertion', 'application'],
            'benefit': ['advantage', 'entitlement', 'privilege', 'compensation']
        }
        
        expanded_queries = [query]
        
        for term, synonyms in legal_expansions.items():
            if term in query.lower():
                for synonym in synonyms:
                    expanded_query = query.lower().replace(term, synonym)
                    if expanded_query != query.lower():
                        expanded_queries.append(expanded_query)
        
        # Add domain-specific context
        domain_contexts = [
            f"legal document {query}",
            f"policy regarding {query}",
            f"insurance terms {query}" if 'insurance' in query.lower() else f"HR policy {query}"
        ]
        
        expanded_queries.extend(domain_contexts)
        return list(set(expanded_queries))[:5]  # Limit to 5 variations

class ResourceManager:
    """Manage resources and prevent memory leaks"""
    
    @staticmethod
    def cleanup_temp_file(temp_file: Optional[str]) -> None:
        """Safely cleanup temporary files"""
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                logger.info(f"🧹 Cleaned temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"⚠️ Cleanup warning: {e}")
    
    @staticmethod
    def force_gc() -> None:
        """Force garbage collection"""
        collected = gc.collect()
        logger.info(f"🧹 Garbage collected: {collected} objects")
    
    @staticmethod
    def log_memory_usage() -> None:
        """Log current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"📊 Memory usage: {memory_mb:.2f} MB")
        except ImportError:
            logger.info("📊 Memory monitoring unavailable (psutil not installed)")

# Pinecone setup with error handling
try:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
    
    logger.info(f"🔧 Initializing Pinecone index: {INDEX_NAME}")
    
    if not pc.has_index(INDEX_NAME):
        logger.info("🔧 Creating new Pinecone index...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    index = pc.Index(INDEX_NAME)
    logger.info("✅ Pinecone initialized successfully")
    
except Exception as e:
    logger.error(f"❌ Pinecone initialization failed: {e}")
    raise

# Enhanced model configurations
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    chunk_size=500,
    max_retries=2,
    request_timeout=20,
    dimensions=3072  # Use full dimensions for better accuracy
)

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=800,  # Increased for more detailed responses
    timeout=25,  # Slightly increased for better responses
    max_retries=2
)

def generate_document_id(url: str) -> str:
    """Generate document ID from URL"""
    doc_id = hashlib.md5(url.encode()).hexdigest()[:16]
    logger.info(f"📋 Generated document ID: {doc_id}")
    return doc_id

def get_document_namespace(doc_id: str) -> str:
    """Get document-specific namespace"""
    namespace = f"doc_{doc_id}"
    logger.debug(f"📋 Using namespace: {namespace}")
    return namespace

async def check_document_exists_async(doc_id: str) -> bool:
    """Check if document exists with enhanced logging"""
    start_time = time.time()
    
    try:
        doc_namespace = get_document_namespace(doc_id)
        loop = asyncio.get_event_loop()
        
        logger.info(f"🔍 Checking document existence: {doc_id} in {doc_namespace}")
        
        # Create a dummy vector for existence check
        dummy_vector = [0.1] * 3072
        
        result = await loop.run_in_executor(
            None,
            lambda: index.query(
                vector=dummy_vector,
                top_k=1,
                namespace=doc_namespace,
                include_metadata=False
            )
        )
        
        exists = len(result.matches) > 0
        check_time = time.time() - start_time
        
        logger.info(f"🔍 Document {doc_id} exists in namespace {doc_namespace}: {exists} (took {check_time:.2f}s)")
        return exists
        
    except Exception as e:
        logger.error(f"❌ Error checking document existence: {e}")
        return False

async def download_pdf_async(url: str) -> str:
    """Enhanced PDF download with proper error handling"""
    start_time = time.time()
    temp_file = None
    
    try:
        logger.info(f"📥 Starting PDF download: {url[:100]}...")
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise Exception(f"HTTP {resp.status}: {resp.reason}")
                
                content_length = resp.headers.get('content-length')
                if content_length:
                    size_mb = int(content_length) / 1024 / 1024
                    logger.info(f"📥 Downloading PDF: {size_mb:.2f} MB")
                
                content = await resp.read()
                
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            temp_file = tmp.name
            
        download_time = time.time() - start_time
        logger.info(f"✅ PDF downloaded successfully in {download_time:.2f}s: {temp_file}")
        
        return temp_file
        
    except asyncio.TimeoutError:
        if temp_file:
            ResourceManager.cleanup_temp_file(temp_file)
        raise Exception("Download timeout - PDF too large or slow connection")
    except Exception as e:
        if temp_file:
            ResourceManager.cleanup_temp_file(temp_file)
        logger.error(f"❌ Download failed: {e}")
        raise Exception(f"Download error: {e}")

def clean_and_structure_text(text: str) -> str:
    """Enhanced text cleaning with structure preservation"""
    # Remove excessive whitespace but preserve paragraph structure
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double
    text = re.sub(r' +', ' ', text)  # Multiple spaces to single
    
    # Clean up special characters while preserving important punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\%\$\'\"\n\[\]]', ' ', text)
    
    # Fix common PDF extraction issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Fix merged words
    text = re.sub(r'(\w)(\d+)', r'\1 \2', text)  # Separate words from numbers
    
    return text.strip()

def create_enhanced_chunks(temp_file: str) -> List[Document]:
    """Create enhanced chunks with better structure awareness"""
    start_time = time.time()
    
    try:
        logger.info(f"📄 Loading PDF: {temp_file}")
        
        loader = PyPDFLoader(temp_file)
        docs = loader.load()
        
        if not docs:
            raise Exception("No content extracted from PDF")
        
        logger.info(f"📄 Loaded {len(docs)} pages from PDF")
        
        # Combine pages with better structure preservation
        full_text = ""
        for i, doc in enumerate(docs):
            page_content = clean_and_structure_text(doc.page_content)
            full_text += f"\n\n--- PAGE {i+1} ---\n\n{page_content}"
        
        if len(full_text) < 200:
            raise Exception("PDF content too short - possibly corrupted")
        
        logger.info(f"📄 Total text length: {len(full_text)} characters")
        
        # Detect document structure
        processor = EnhancedTextProcessor()
        doc_structure = processor.detect_document_structure(full_text)
        
        # Enhanced chunking strategy
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Increased for better context retention
            chunk_overlap=100,  # Increased overlap for better continuity
            separators=[
                "\n\n--- PAGE", "\n\n", "\n", ". ", "; ", ", ", " ", ""
            ],
            length_function=len,
            is_separator_regex=False,
        )
        
        # Create document with enhanced metadata
        full_doc = Document(
        page_content=full_text,
       metadata={
           "source": temp_file,
           "total_pages": len(docs),
           "headers": doc_structure.get("headers", [])
           # dropped `structure` dict (unsupported type)
        }
    )
        
        chunks = splitter.split_documents([full_doc])
        
        # Enhanced chunk processing
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            content = chunk.page_content.strip()
            
            # Skip very short chunks
            if len(content) < 100:
                continue
            
            # Calculate importance score
            importance = processor.calculate_importance_score(content, doc_structure)
            
            # Determine chunk type
            chunk_type = "paragraph"
            if any(header in content for header in doc_structure.get('headers', [])):
                chunk_type = "header_section"
            elif any(content.startswith(list_item[1][:10]) for list_item in doc_structure.get('lists', [])):
                chunk_type = "list"
            elif '|' in content and content.count('|') >= 2:
                chunk_type = "table"
            
            # Create enhanced metadata
            chunk.metadata.update({
                "chunk_id": f"chunk_{i}_{int(time.time())}",
                "chunk_index": i,
                "chunk_length": len(content),
                "word_count": len(content.split()),
                "chunk_type": chunk_type,
                "importance_score": importance,
                "text": content[:45000],  # Increased for Pinecone metadata
                "preview": content[:200] + "..." if len(content) > 200 else content
            })
            
            enhanced_chunks.append(chunk)
        
        # Sort by importance to prioritize better chunks
        enhanced_chunks.sort(key=lambda x: x.metadata['importance_score'], reverse=True)
        
        process_time = time.time() - start_time
        logger.info(f"📄 Created {len(enhanced_chunks)} enhanced chunks in {process_time:.2f}s")
        
        # Memory cleanup
        del docs, full_text, chunks
        ResourceManager.force_gc()
        
        return enhanced_chunks
        
    except Exception as e:
        logger.error(f"❌ Enhanced document processing failed: {e}")
        raise

async def enhanced_batch_embed_queries(questions: List[str]) -> Dict[str, List[float]]:
    """Enhanced batch embedding with query expansion"""
    start_time = time.time()
    
    # Expand queries
    expander = QueryExpander()
    all_queries = []
    query_mapping = {}
    
    for question in questions:
        expanded = expander.expand_legal_query(question)
        all_queries.extend(expanded)
        query_mapping[question] = expanded
    
    logger.info(f"🔍 Expanded {len(questions)} queries to {len(all_queries)} variations")
    
    # Check cache and embed
    cached_embeddings = {}
    uncached_queries = []
    
    for query in all_queries:
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if query_hash in EMBEDDING_CACHE:
            cached_embeddings[query] = EMBEDDING_CACHE[query_hash]
        else:
            uncached_queries.append(query)
    
    logger.info(f"🎯 Cache hits: {len(cached_embeddings)}, New embeddings: {len(uncached_queries)}")
    
    if uncached_queries:
        try:
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                embedding_results = await loop.run_in_executor(
                    executor, embeddings.embed_documents, uncached_queries
                )
            
            # Cache new embeddings
            for query, emb in zip(uncached_queries, embedding_results):
                query_hash = hashlib.md5(query.encode()).hexdigest()
                EMBEDDING_CACHE[query_hash] = emb
                cached_embeddings[query] = emb
                
        except Exception as e:
            logger.error(f"❌ Enhanced embedding generation failed: {e}")
            raise
    
    # Combine embeddings for original questions
    final_embeddings = {}
    for original_question in questions:
        expanded_queries = query_mapping[original_question]
        # Use the first embedding (original query) as primary
        if expanded_queries[0] in cached_embeddings:
            final_embeddings[original_question] = cached_embeddings[expanded_queries[0]]
    
    embed_time = time.time() - start_time
    logger.info(f"✅ Enhanced embedding completed in {embed_time:.2f}s")
    
    return final_embeddings

async def hybrid_retrieve(question: str, doc_id: str, k: int = 8) -> List[Document]:
    """Hybrid retrieval with reranking"""
    start_time = time.time()
    
    try:
        doc_namespace = get_document_namespace(doc_id)
        
        logger.info(f"🔍 Hybrid search in namespace: {doc_namespace}")
        
        # Get enhanced embeddings
        question_embeddings = await enhanced_batch_embed_queries([question])
        if question not in question_embeddings:
            return []
            
        question_embedding = question_embeddings[question]
        
        loop = asyncio.get_event_loop()
        
        # Retrieve more candidates for reranking
        result = await loop.run_in_executor(
            None,
            lambda: index.query(
                vector=question_embedding,
                top_k=k * 3,  # Retrieve 3x more for better reranking
                namespace=doc_namespace,
                include_metadata=True
            )
        )
        
        logger.info(f"📊 Retrieved {len(result.matches)} candidates for reranking")
        
        if not result.matches:
            return []
        
        # Enhanced reranking based on multiple factors
        scored_matches = []
        for match in result.matches:
            if not match.metadata or "text" not in match.metadata:
                continue
                
            text = match.metadata["text"]
            
            # Multiple scoring factors
            semantic_score = match.score
            importance_score = match.metadata.get('importance_score', 1.0)
            chunk_type_score = {
                'header_section': 1.3,
                'list': 1.1,
                'table': 1.2,
                'paragraph': 1.0
            }.get(match.metadata.get('chunk_type', 'paragraph'), 1.0)
            
            # Keyword relevance score
            question_words = set(question.lower().split())
            text_words = set(text.lower().split())
            keyword_overlap = len(question_words.intersection(text_words)) / len(question_words)
            
            # Combined score
            final_score = (semantic_score * 0.5 + 
                         importance_score * 0.2 + 
                         chunk_type_score * 0.15 + 
                         keyword_overlap * 0.15)
            
            scored_matches.append((match, final_score))
        
        # Sort by combined score
        scored_matches.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to documents
        docs = []
        for match, final_score in scored_matches[:k]:
            doc = Document(
                page_content=match.metadata["text"],
                metadata={
                    **match.metadata,
                    "semantic_score": match.score,
                    "final_score": final_score
                }
            )
            docs.append(doc)
        
        search_time = time.time() - start_time
        scores = [f'{d.metadata["final_score"]:.3f}' for d in docs[:3]]
        logger.info(f"✅ Hybrid retrieval completed in {search_time:.2f}s, top scores: {scores}")
        
        return docs
        
    except Exception as e:
        logger.error(f"❌ Hybrid retrieval failed: {e}")
        return []

def build_enhanced_rag_chain(splits: List[Document], doc_id: str):
    """Build enhanced RAG chain with improved accuracy"""
    doc_namespace = get_document_namespace(doc_id)
    
    vector_store = PineconeVectorStore(
        index=index, 
        embedding=embeddings, 
        namespace=doc_namespace
    )

    async def upload_enhanced_chunks():
        """Upload enhanced chunks with better metadata"""
        if not splits:
            return
            
        upload_start = time.time()
        logger.info(f"📤 Uploading {len(splits)} enhanced chunks")
        
        try:
            # Prepare enhanced documents
            docs_to_upload = []
            vector_ids = []
            
            for i, split in enumerate(splits):
                vector_id = f"enhanced_chunk_{i}_{int(time.time())}"
                vector_ids.append(vector_id)
                
                # Enhanced metadata for better retrieval
                split.metadata.update({
                    "vector_id": vector_id,
                    "upload_timestamp": int(time.time()),
                    "enhanced_chunk": True
                })
                docs_to_upload.append(split)
            
            # Upload in optimized batches
            batch_size = 10  # Smaller batches for stability
            for i in range(0, len(docs_to_upload), batch_size):
                batch_docs = docs_to_upload[i:i+batch_size]
                batch_ids = vector_ids[i:i+batch_size]
                
                vector_store.add_documents(
                    documents=batch_docs,
                    ids=batch_ids
                )
                
                await asyncio.sleep(0.2)  # Slight delay for stability
            
            upload_time = time.time() - upload_start
            logger.info(f"✅ Enhanced upload completed in {upload_time:.2f}s")
            
            # Verification
            await asyncio.sleep(2)  # Wait for indexing
            test_result = index.query(
                vector=[0.1] * 3072,
                top_k=5,
                namespace=doc_namespace,
                include_metadata=True
            )
            
            logger.info(f"✅ Verified {len(test_result.matches)} chunks indexed")
            
        except Exception as e:
            logger.error(f"❌ Enhanced upload failed: {e}")
            raise
        
        PROCESSED_DOCS[doc_id] = True

    # Upload if needed
    if splits:
        asyncio.create_task(upload_enhanced_chunks())

    async def enhanced_rag_pipeline(questions: List[str]) -> List[str]:
        """Enhanced RAG pipeline with improved accuracy"""
        pipeline_start = time.time()
        logger.info(f"🔍 Enhanced RAG processing {len(questions)} questions")
        
        if splits:
            await upload_enhanced_chunks()
        
        try:
            # Parallel retrieval for better performance
            retrieval_tasks = []
            for question in questions:
                task = hybrid_retrieve(question, doc_id, k=6)
                retrieval_tasks.append(task)
            
            all_docs = await asyncio.gather(*retrieval_tasks)
            
            # Generate enhanced answers
            answers = []
            prompt = build_chat_prompt()
            
            for i, (question, docs) in enumerate(zip(questions, all_docs)):
                answer_start = time.time()
                
                try:
                    if docs:
                        # Build enhanced context
                        context_parts = []
                        for j, doc in enumerate(docs[:4]):
                            score = doc.metadata.get('final_score', 0)
                            chunk_type = doc.metadata.get('chunk_type', 'paragraph')
                            content = doc.page_content[:1500]  # Increased context
                            
                            context_parts.append(
                                f"[Relevant Section {j+1} - Type: {chunk_type}, Relevance: {score:.3f}]:\n{content}\n"
                            )
                        
                        context = "\n".join(context_parts)
                    else:
                        context = "No relevant information found in the document."
                    
                    # Enhanced prompt formatting
                    formatted_prompt = prompt.format_messages(
                        context=context, 
                        question=f"Based on the provided document context, please answer: {question}"
                    )
                    
                    # Generate response
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None, 
                        lambda: llm.invoke(formatted_prompt)
                    )
                    
                    result = response.content.strip()
                    
                    # Enhanced result processing
                    if result and not result.endswith(('.', '!', '?')):
                        result += "."
                    
                    answer_time = time.time() - answer_start
                    logger.info(f"✅ Enhanced Q{i + 1} answered in {answer_time:.2f}s")
                    
                    answers.append(result)
                    
                except Exception as e:
                    logger.error(f"❌ Error in enhanced processing Q{i + 1}: {e}")
                    answers.append("I apologize, but I encountered an error processing this question.")
            
            pipeline_time = time.time() - pipeline_start
            logger.info(f"✅ Enhanced RAG pipeline completed in {pipeline_time:.2f}s")
            
            return answers
            
        except Exception as e:
            logger.error(f"❌ Enhanced RAG pipeline failed: {e}")
            return [f"Error: {e}"] * len(questions)

    return enhanced_rag_pipeline

async def process_query_enhanced(pdf_url: str, questions: List[str]) -> List[str]:
    """Enhanced main processing function"""
    process_start = time.time()
    doc_id = generate_document_id(pdf_url)
    temp_file = None
    
    try:
        logger.info(f"📋 Enhanced processing for {len(questions)} questions")
        logger.info(f"📋 Document ID: {doc_id}")
        
        ResourceManager.log_memory_usage()
        
        # Check if processing needed
        needs_processing = not (doc_id in PROCESSED_DOCS or await check_document_exists_async(doc_id))
        
        splits = []
        if needs_processing:
            logger.info(f"📥 Enhanced document processing required")
            
            # Download and process
            temp_file = await download_pdf_async(pdf_url)
            
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                splits = await loop.run_in_executor(executor, create_enhanced_chunks, temp_file)
        
        # Build and run enhanced RAG
        rag_chain = build_enhanced_rag_chain(splits, doc_id)
        answers = await rag_chain(questions)
        
        total_time = time.time() - process_start
        logger.info(f"✅ Enhanced processing completed in {total_time:.2f}s")
        
        return answers

    except Exception as e:
        logger.error(f"❌ Enhanced processing failed: {e}")
        return [f"Enhanced processing error: {str(e)[:100]}..."] * len(questions)

    finally:
        if temp_file:
            ResourceManager.cleanup_temp_file(temp_file)
        ResourceManager.force_gc()
        logger.info("🧹 Enhanced cleanup completed")