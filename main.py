import os, tempfile, requests, asyncio
from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.rag_engine import process_query
from typing import Optional

app = FastAPI()
API_TOKEN = os.environ["HACKRX_API_TOKEN"]

def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid auth header")
    token = authorization.split()[1]
    if token != API_TOKEN:
        raise HTTPException(403, "Invalid token")

class QueryRequest(BaseModel):
    documents: str
    questions: list[str]

@app.post("/hackrx/run")
async def run_rag(payload: QueryRequest, auth=Depends(verify_token)):
    # Download PDF
    resp = requests.get(payload.documents)
    resp.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(resp.content); tmp.close()

    try:
        answers = await process_query(tmp.name, payload.questions)
        return JSONResponse({"answers": answers})
    finally:
        os.remove(tmp.name)
