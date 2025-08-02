# app/utils.py
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
import os 
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)
# app/utils.py
# import your classifier prompt text


# llm = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=openai_key)

# _domain_template = PromptTemplate(
#     template=DOMAIN_CLASSIFIER + "\n\nQuery: {query}\n\nAnswer with one domain:",
#     input_variables=["query"]
# )
# domain_chain = _domain_template | llm
# def detect_domain_from_query(query: str) -> str:
#     """Run a small LLMChain to pick one of [insurance, legal, hr, compliance]."""
#     raw = domain_chain.invoke({"query": query})
#     return raw.content.strip().lower()  # ✅ .content is the string message

