import os
import asyncio
from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from pydantic import BaseModel
import aiohttp
import time
from app.rag_vector import OptimizedVectorRAG  # Updated import
app = FastAPI(
    title="HackRx 6.0 RAG API", 
    version="1.0.0",
    timeout=30  # Set global timeout
)

# Environment variables
API_TOKEN = os.environ["HACKRX_API_TOKEN"]

# Global warmup state and RAG instance
SYSTEM_WARMED = False
rag_system = None

async def initialize_rag():
    """Initialize RAG system once"""
    global rag_system
    if rag_system is None:
        print("🚀 Initializing RAG system...")
        rag_system = OptimizedVectorRAG()
        print("✅ RAG system initialized successfully")
    return rag_system

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

async def process_query_enhanced(document_url: str, questions: list[str]) -> list[str]:
    """
    Enhanced query processing using OptimizedVectorRAG
    """
    try:
        # Get RAG system instance
        rag = await initialize_rag()
        
        # Generate document ID from URL
        doc_id = rag.generate_doc_id(document_url)
        print(f"📄 Document ID: {doc_id}")
        
        # Check if document exists, if not process it
        doc_exists = await rag.check_document_exists(doc_id)
        if not doc_exists:
            print(f"📥 Processing new document: {document_url}")
            processed_doc_id = await rag.process_document(document_url, doc_id)
            print(f"✅ Document processed: {processed_doc_id}")
        else:
            print(f"📋 Using cached document: {doc_id}")
        
        # Process all questions
        answers = []
        for i, question in enumerate(questions, 1):
            print(f"❓ Processing question {i}/{len(questions)}: {question[:50]}...")
            
            try:
                answer = await rag.query(question, doc_id)
                answers.append(answer)
                print(f"✅ Answer {i} generated successfully")
                
            except Exception as e:
                print(f"❌ Error processing question {i}: {e}")
                # Provide fallback answer instead of failing completely
                answers.append(f"Sorry, I couldn't process this question: {str(e)}")
        
        print(f"🎯 Successfully processed {len(answers)} questions")
        return answers
        
    except Exception as e:
        print(f"❌ Enhanced processing error: {e}")
        # Return error messages for all questions instead of raising
        error_msg = f"Error processing document: {str(e)}"
        return [error_msg for _ in questions]

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_rag(payload: QueryRequest, auth=Depends(verify_token)):
    """
    Optimized RAG endpoint for document Q&A
    Processes documents and answers questions using cached vectors
    """
    start_time = time.time()
    
    try:
        # Input validation
        if not payload.documents or not payload.questions:
            raise HTTPException(400, "Documents URL and questions are required")
        
        if len(payload.questions) > 15:  # Reduced limit for faster processing
            raise HTTPException(400, "Too many questions (max 15)")
        
        # Log request info
        print(f"📋 Processing {len(payload.questions)} questions for document")
        
        # Directly await without timeout
        answers = await process_query_enhanced(payload.documents, payload.questions)
        
        processing_time = time.time() - start_time
        print(f"⏱️ Total processing time: {processing_time:.2f}s")
        
        return QueryResponse(answers=answers)
        
    except aiohttp.ClientError as e:
        raise HTTPException(400, f"Failed to access document: {str(e)}")
    except asyncio.TimeoutError:
        raise HTTPException(408, "Request timeout")
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ Processing error after {processing_time:.2f}s: {e}")
        raise HTTPException(500, f"Internal processing error: {str(e)}")

# Optional: Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        rag = await initialize_rag()
        return {
            "status": "healthy",
            "rag_system": "initialized",
            "service": "HackRx 6.0 RAG API"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "service": "HackRx 6.0 RAG API"
        }

# Optional: Warmup endpoint
@app.post("/warmup")
async def warmup_system(background_tasks: BackgroundTasks):
    """Warmup the RAG system"""
    global SYSTEM_WARMED
    try:
        await initialize_rag()
        SYSTEM_WARMED = True
        return {"status": "warmed", "message": "RAG system initialized"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Add middleware for request timing
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        timeout_keep_alive=30,
    )