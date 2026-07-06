from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver



from backend.agents.tools import (
    search_arxiv,
    search_internal_docs,
    search_uploaded_document,
    calculator
)

load_dotenv()

llm = ChatMistralAI(model="mistral-large-latest")

web_search = TavilySearch(max_results =3)

memory = MemorySaver()

SYSTEM_PROMPT = """You are an enterprise research assistant with access to these tools:
search_arxiv, search_internal_docs, search_uploaded_document, calculator, web_search.

Tool selection rules:
- If the user refers to "the document", "the PDF", "this file", "chapter X", "the uploaded
  file", or anything implying a document uploaded in this session, ALWAYS call
  search_uploaded_document FIRST. Do not ask a clarifying question before trying it.
- Only ask the user to clarify if search_uploaded_document explicitly returns
  "No document has been uploaded in this session yet."
- If search_uploaded_document finds no relevant content for the query, fall back to
  search_internal_docs before giving up.
- For questions about academic papers or "recent research", use search_arxiv.
- For current events, prices, or anything time-sensitive, use web_search.
- For math expressions, use calculator.
- Always cite the source file name shown in tool results when answering from documents.
"""

agent = create_agent(
    llm,
    tools=[
        search_arxiv,
        search_internal_docs,
        search_uploaded_document,
        calculator,
        web_search
    ],
    checkpointer=memory,
    system_prompt=SYSTEM_PROMPT
)