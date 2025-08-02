import os
import asyncio
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from .prompt import build_chat_prompt          # imports updated prompt
from .utils  import format_docs                # helper to format retrieved docs

# Load environment variables
load_dotenv()

# Pinecone setup
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
NAMESPACE = "hackrx"

if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # for OpenAI's `text-embedding-3-small`
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)

# LangChain components
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm        = ChatOpenAI(model="gpt-3.5-turbo-0125")

def build_rag_chain(splits):
    # Ingest to Pinecone
    vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace=NAMESPACE)
    vector_store.add_documents(splits)

    # Create retriever
    retriever = vector_store.as_retriever()

    # Build prompt
    prompt = build_chat_prompt()

    # RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

async def process_query(pdf_path: str, questions: list[str]) -> list[str]:
    # Load and split document
    loader   = PyPDFLoader(pdf_path)
    docs     = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits   = splitter.split_documents(docs)

    # Build chain
    rag_chain = build_rag_chain(splits)

    # Process all 10 questions concurrently in a single chain call
    return await asyncio.gather(*[rag_chain.ainvoke(q) for q in questions])
