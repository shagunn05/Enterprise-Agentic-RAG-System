"""
backend/reports/report_generator.py
-------------------------------------
Takes retrieved documents and generates a structured report using Mistral AI.

Usage:
    from backend.reports.report_generator import generate_report

    report_text = generate_report(
        query="Give an overview of the education system",
        documents=retrieved_docs   # list of LangChain Document objects
    )
    print(report_text)
"""

import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # Loads MISTRAL_API_KEY from .env

# -----------------------------
# LLM SETUP
# -----------------------------
llm = ChatMistralAI(
    model="mistral-large-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.3,   # low creativity for factual, report-style output
)

# -----------------------------
# PROMPT TEMPLATE
# -----------------------------
REPORT_PROMPT = ChatPromptTemplate.from_template("""
You are a professional research analyst. Based on the documents provided below,
generate a structured report that answers the user's query.

User Query: {query}

Documents:
{context}

Format the report as follows:

# Report Title

## Executive Summary
(2-3 line overall summary)

## Key Findings
- point 1
- point 2
- point 3 (as many relevant points as needed)

## Detailed Analysis
(Organize and elaborate on the document content in paragraph form)

## Conclusion
(1-2 line closing summary)

Rules:
- Only use information found in the provided documents; do not invent facts.
- If the documents don't contain enough information, clearly state that.
- Use clear, professional language.
""")

# -----------------------------
# CHAIN
# -----------------------------
report_chain = REPORT_PROMPT | llm | StrOutputParser()

def _format_documents(documents) -> str:
    """Handles both LangChain Document objects and plain text strings."""
    formatted = []
    for i, doc in enumerate(documents, start=1):
        if hasattr(doc, "page_content"):
            source = doc.metadata.get("source", "Unknown source")
            formatted.append(f"[Document {i} — {source}]\n{doc.page_content}\n")
        else:
            # Plain string output (e.g. from the arxiv retriever)
            formatted.append(f"[Source {i}]\n{doc}\n")
    return "\n---\n".join(formatted)

def generate_report(query: str, documents: list) -> str:
    """
    Generates a structured report string from a query and retrieved documents.

    Args:
        query: The original user question / report topic
        documents: List of LangChain Document objects (from a retriever)

    Returns:
        Markdown-formatted report (string)
    """
    if not documents:
        return "⚠️ No documents were retrieved. Run the retriever first, then generate the report."

    context = _format_documents(documents)

    report = report_chain.invoke({
        "query": query,
        "context": context,
    })

    return report


def save_report_to_file(report_text: str, filename: str = "report.md") -> str:
    """Saves the report inside the backend/reports/generated/ folder."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "generated")
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_text)

    return filepath


# -----------------------------
# QUICK TEST (for running this file directly)
# -----------------------------
if __name__ == "__main__":
    from langchain_core.documents import Document

    # Dummy test documents — replace with your actual retriever output
    test_docs = [
        Document(
            page_content="AI adoption in education is growing rapidly, "
                          "especially in personalized learning and automated grading.",
            metadata={"source": "education.pdf", "page": 1},
        ),
        Document(
            page_content="RAG-based tutoring systems answer students' questions "
                          "in real time using relevant course material.",
            metadata={"source": "education.pdf", "page": 2},
        ),
    ]

    result = generate_report("What is the role of AI in education?", test_docs)
    print(result)

    path = save_report_to_file(result, "test_report.md")
    print(f"\n✅ Report saved at: {path}")