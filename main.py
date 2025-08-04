import os
import asyncio
from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from pydantic import BaseModel
from app.rag_engine import process_query
import aiohttp
import time

app = FastAPI(
    title="HackRx 6.0 RAG API", 
    version="1.0.0",
    timeout=30  # Set global timeout
)

# Environment variables
API_TOKEN = os.environ["HACKRX_API_TOKEN"]

# Global warmup state
SYSTEM_WARMED = False

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
        
        # Set timeout for the entire operation
        try:
            answers = await asyncio.wait_for(
                process_query(payload.documents, payload.questions), timeout=50
            )
        except asyncio.TimeoutError:
            raise HTTPException(408, "Request timeout - processing took too long")
        
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
        timeout_notify=25
    )