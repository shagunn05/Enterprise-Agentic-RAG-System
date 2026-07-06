from  langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
import streamlit as  st

st.write("loading vector store")

load_dotenv()

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
)

vectorstore = Chroma(
   
    embedding_function = embeddings,
    persist_directory="chroma-db"
)
