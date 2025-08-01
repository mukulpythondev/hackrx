# app/rag_engine.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .prompt import build_chat_prompt          # << import here
from .utils    import format_docs               # << helper function

load_dotenv()

# Pinecone & index setup (same as before)
pc          = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
INDEX_NAME  = os.environ["PINECONE_INDEX_NAME"]
NAMESPACE   = "hackrx"
if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # Required for OpenAI's `text-embedding-3-small`
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(INDEX_NAME)

# LLM setup
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm        = ChatOpenAI(model="gpt-3.5-turbo-0125")

def build_rag_chain(splits, domain: str):
    # ingest into Pinecone
    vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace=NAMESPACE)
    vector_store.add_documents(splits)

    # build prompt for this domain
    prompt = build_chat_prompt(domain)

    retriever = vector_store.as_retriever()
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def process_query(pdf_path: str, questions: list[str], domain: str) -> list[str]:
    # load & split
    loader   = PyPDFLoader(pdf_path)
    docs     = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits   = splitter.split_documents(docs)

    # build & run chain
    rag_chain = build_rag_chain(splits, domain)
    return [rag_chain.invoke(q) for q in questions]
