

from backend.vector_store.DB import vectorstore 


mmr_retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs ={"k":3,"fetch_k":10}
)
