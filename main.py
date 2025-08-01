# app/main.py
import os, tempfile, requests
from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.utils import detect_domain_from_query
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
    domain: Optional[str] = None

@app.post("/hackrx/run")
async def run_rag(payload: QueryRequest, auth=Depends(verify_token)):
    # download PDF
    resp = requests.get(payload.documents)
    resp.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(resp.content); tmp.close()
    domain = payload.domain
    if domain is None:
        # pick the first question to classify
        domain = detect_domain_from_query(payload.questions[0])
    try:
        answers = process_query(tmp.name, payload.questions, payload.domain)
        return JSONResponse({"answers": answers})
    finally:
        os.remove(tmp.name)
