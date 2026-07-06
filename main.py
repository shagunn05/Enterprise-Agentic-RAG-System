from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from backend.agents.agent import agent
from backend.retrievers.mmr import mmr_retriever
from backend.retrievers.multiquery import multiquery_retriever
from backend.retrievers.arxiv_retrivers import get_arxiv_retriever
from pathlib import Path
import subprocess
import sys

from backend.reports.report_generator import generate_report, save_report_to_file
load_dotenv()

#create database automatically if it doesnot exist
if not Path("chroma-db").exists():
    print("chroma-db not found. creating database...")
    subprocess.run([sys.executable,"create_database.py"],
                   check=True)
    

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
)

vectorstore = Chroma(
    persist_directory= "chroma-db",
    embedding_function = embeddings,
)


query = "what is the role of ai in education system?"


mmr_docs = mmr_retriever.invoke(query)              # LangChain Document objects
multiquery_docs = multiquery_retriever.invoke(query)  # LangChain Document objects
arxiv_text = get_arxiv_retriever(query, max_results=3)  # plain string


combined_sources = mmr_docs + multiquery_docs + [arxiv_text]


report = generate_report(query, combined_sources)
print(report)

path = save_report_to_file(report, "combined_report.md")
print(f"\n✅ Report saved at: {path}")

 