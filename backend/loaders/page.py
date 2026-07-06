from langchain_community.document_loaders import WebBaseLoader

url = "https://docs.x.ai/overview"

data = WebBaseLoader(url)

docs = data.load()

print(docs[0].page_content)