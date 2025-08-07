import os
import asyncio
import hashlib
from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from pydantic import BaseModel
from app.rag_graph import HybridRAGSystem  # Updated import
import aiohttp
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HackRx 6.0 Hybrid RAG API", 
    version="1.0.0",
    timeout=30
)

# Environment variables
API_TOKEN = os.environ["HACKRX_API_TOKEN"]

# Global RAG system instance
rag_system: HybridRAGSystem = None
SYSTEM_WARMED = False

@app.on_event("startup")
async def startup_event():
    """Initialize the hybrid RAG system on startup"""
    global rag_system, SYSTEM_WARMED
    try:
        logger.info("🚀 Initializing Hybrid RAG System...")
        rag_system = HybridRAGSystem()
        SYSTEM_WARMED = True
        logger.info("✅ Hybrid RAG System initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    global rag_system
    if rag_system:
        rag_system.close()
        logger.info("✅ RAG system closed")

def verify_token(authorization: str = Header(...)):
    """Verify Bearer token authentication"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid authorization header format")
    
    token = authorization.split(" ", 1)[1]
    if token != API_TOKEN:
        raise HTTPException(403, "Invalid API token")
    
    return token

class QueryRequest(BaseModel):
    documents: str
    questions: list[str]

class QueryResponse(BaseModel):
    answers: list[str]

async def process_document_and_questions(documents_url: str, questions: list[str]) -> list[str]:
    """
    Process document using hybrid RAG and answer questions
    """
    global rag_system
    
    if not rag_system:
        raise HTTPException(500, "RAG system not initialized")
    
    try:
        # Generate document ID from URL
        doc_id = hashlib.md5(documents_url.encode()).hexdigest()[:12]
        
        logger.info(f"📋 Processing document: {doc_id}")
        logger.info(f"🔍 Questions to answer: {len(questions)}")
        
        # Check if document already exists in our systems
        doc_exists = await check_document_exists(doc_id)
        
        if not doc_exists:
            logger.info("📥 Document not found, processing...")
            # Process new document (store in both vector and graph DB)
            doc_id = await rag_system.process_document(documents_url, doc_id)
            logger.info(f"✅ Document processed and stored: {doc_id}")
        else:
            logger.info("📋 Document already exists, using cached version")
        
        # Answer questions using hybrid search
        answers = []
        for i, question in enumerate(questions, 1):
            logger.info(f"🔍 Processing question {i}/{len(questions)}")
            answer = await rag_system.query(question, doc_id)
            answers.append(answer)
        
        return answers
        
    except Exception as e:
        logger.error(f"❌ Processing error: {e}")
        raise

async def check_document_exists(doc_id: str) -> bool:
    """
    Check if document exists in both vector and graph databases
    """
    try:
        # Quick check in Pinecone (vector DB)
        dummy_vector = [0.1] * 1536  # Adjust dimension as needed
        result = rag_system.index.query(
            vector=dummy_vector,
            top_k=1,
            filter={'doc_id': doc_id},
            include_metadata=False
        )
        
        vector_exists = len(result.matches) > 0
        
        # Quick check in Neo4j (graph DB)
        with rag_system.neo4j_driver.session() as session:
            graph_result = session.run(
                "MATCH (d:Document {id: $doc_id}) RETURN count(d) as count",
                doc_id=doc_id
            )
            graph_exists = graph_result.single()['count'] > 0
        
        exists = vector_exists and graph_exists
        logger.info(f"📋 Document {doc_id} exists - Vector: {vector_exists}, Graph: {graph_exists}")
        
        return exists
        
    except Exception as e:
        logger.warning(f"⚠️ Error checking document existence: {e}")
        return False

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_hybrid_rag(payload: QueryRequest, auth=Depends(verify_token)):
    """
    Hybrid RAG endpoint for document Q&A
    Uses both vector search and knowledge graph for enhanced accuracy
    """
    start_time = time.time()
    
    try:
        # System readiness check
        if not SYSTEM_WARMED or not rag_system:
            raise HTTPException(503, "RAG system not ready")
        
        # Input validation
        if not payload.documents or not payload.questions:
            raise HTTPException(400, "Documents URL and questions are required")
        
        if len(payload.questions) > 15:
            raise HTTPException(400, "Too many questions (max 15)")
        
        # Validate URL format
        if not payload.documents.startswith(('http://', 'https://')):
            raise HTTPException(400, "Invalid document URL format")
        
        # Log request info
        logger.info(f"📋 Hybrid RAG request: {len(payload.questions)} questions")
        logger.info(f"📄 Document URL: {payload.documents[:100]}...")
        
        # Process using hybrid approach
        answers = await process_document_and_questions(
            payload.documents, 
            payload.questions
        )
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Hybrid processing completed in {processing_time:.2f}s")
        
        return QueryResponse(answers=answers)
        
    except aiohttp.ClientError as e:
        raise HTTPException(400, f"Failed to access document: {str(e)}")
    except asyncio.TimeoutError:
        raise HTTPException(408, "Request timeout - document too large or complex")
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Hybrid RAG error after {processing_time:.2f}s: {e}")
        raise HTTPException(500, f"Internal processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global rag_system, SYSTEM_WARMED
    
    status = {
        "status": "healthy" if SYSTEM_WARMED and rag_system else "unhealthy",
        "system_warmed": SYSTEM_WARMED,
        "rag_system_ready": rag_system is not None
    }
    
    # Test database connections
    try:
        if rag_system:
            # Test Neo4j connection
            with rag_system.neo4j_driver.session() as session:
                session.run("RETURN 1")
            status["neo4j_connected"] = True
            
            # Test Pinecone connection  
            rag_system.index.describe_index_stats()
            status["pinecone_connected"] = True
        else:
            status["neo4j_connected"] = False
            status["pinecone_connected"] = False
            
    except Exception as e:
        logger.warning(f"⚠️ Health check warning: {e}")
        status["neo4j_connected"] = False
        status["pinecone_connected"] = False
    
    if status["status"] == "unhealthy":
        raise HTTPException(503, detail=status)
    
    return status

@app.get("/stats")
async def get_stats(auth=Depends(verify_token)):
    """Get system statistics"""
    global rag_system
    
    if not rag_system:
        raise HTTPException(503, "RAG system not ready")
    
    try:
        # Get Pinecone stats
        pinecone_stats = rag_system.index.describe_index_stats()
        
        # Get Neo4j stats
        with rag_system.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (d:Document) 
                OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
                RETURN count(DISTINCT d) as documents, count(c) as chunks
            """)
            neo4j_stats = result.single()
        
        return {
            "vector_db": {
                "total_vectors": pinecone_stats.total_vector_count,
                "dimensions": pinecone_stats.dimension,
                "namespaces": len(pinecone_stats.namespaces)
            },
            "graph_db": {
                "documents": neo4j_stats["documents"],
                "chunks": neo4j_stats["chunks"]
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Stats error: {e}")
        raise HTTPException(500, f"Failed to get stats: {str(e)}")

# Add middleware for request timing and logging
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(f"📨 {request.method} {request.url}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-RAG-Type"] = "hybrid"
    
    # Log response
    logger.info(f"📤 Response: {response.status_code} in {process_time:.2f}s")
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        timeout_keep_alive=30,
        timeout_notify=25
    )