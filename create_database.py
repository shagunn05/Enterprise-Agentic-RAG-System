  # load pdf
 # split into chunks
 # create the embeddings
 # store into chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

# ---- Step 1: Multiple PDFs load karo ----
pdf_folder = "backend/loaders"
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

all_documents = []

for pdf_file in pdf_files:
    file_path = os.path.join(pdf_folder, pdf_file)
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    all_documents.extend(docs)
    print(f"Loaded: {pdf_file} ({len(docs)} pages)")

print(f"\nTotal documents loaded: {len(all_documents)}")

# ---- Step 2: Split into chunks ----
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(all_documents)
print(f"Total chunks created: {len(chunks)}")

# ---- Step 3: Embeddings ----
embeddings = MistralAIEmbeddings(
    model="mistral-embed",
)

# ---- Step 4: Store into Chroma ----
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma-db"
)

vectorstore.persist()

print("\nVector database created successfully with all PDFs!")
print("Database saved successfully")

