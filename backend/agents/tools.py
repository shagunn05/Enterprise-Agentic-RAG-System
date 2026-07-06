

import ast
import math
import operator

from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from backend.retrievers.arxiv_retrivers import get_arxiv_retriever
from backend.retrievers.mmr import mmr_retriever
from backend.retrievers.multiquery import multiquery_retriever
# These should come from your existing retrieval-setup module:
# from your_retrieval_module import get_arxiv_retriever, mmr_retriever, multiquery_retriever


# ==========================================================
# 1) ARXIV SEARCH
# ==========================================================
@tool
def search_arxiv(query: str, recent: bool = False) -> str:
    """Search academic research papers on arxiv. Set recent=True if the user explicitly
    asks for the latest/recent papers, otherwise results are sorted by relevance."""
    return get_arxiv_retriever(query, sort_by_date=recent)


# ==========================================================
# 2) INTERNAL DOCS SEARCH (MMR)
# ==========================================================
@tool
def search_internal_docs(query: str) -> str:
    """Search internal PDF documents using MMR retriever for diverse, relevant results.
    Always cite the source file name(s) shown in the results when answering."""
    docs = mmr_retriever.invoke(query)
    if not docs:
        return "No relevant content found in internal documents."

    output = ""
    for d in docs:
        source = d.metadata.get("source", "unknown source")
        page = d.metadata.get("page", "N/A")
        output += f"\n[Source: {source} | Page: {page}]\n{d.page_content}\n"
    return output


# ==========================================================
# 3) MULTI-ANGLE SEARCH
# ==========================================================
@tool
def search_multi_angle(query: str) -> str:
    """Search internal documents using multiple reworded queries. Use for broad or
    complex questions that a single query might not fully capture."""
    docs = multiquery_retriever.invoke(query)
    if not docs:
        return "No relevant content found."
    output = ""
    for d in docs:
        source = d.metadata.get("source", "unknown source")
        page = d.metadata.get("page", "N/A")
        output += f"\n[Source: {source} | Page: {page}]\n{d.page_content}\n"
    return output


# ==========================================================
# 4) SAFE CALCULATOR (sqrt, log, sin, cos, pi, e, etc.)
# ==========================================================
# Raw eval() is unsafe (sandbox-escape is possible even with the
# __builtins__: {} restriction). So we manually walk the AST and only
# allow a whitelisted set of operations/functions.

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,       # log(x) -> natural log, log(x, base) -> custom base
    "log10": math.log10,
    "log2": math.log2,
    "ln": math.log,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "exp": math.exp,
    "fabs": math.fabs,
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
    "round": round,
    "pow": pow,
}

_ALLOWED_CONSTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINOPS:
            raise ValueError(f"Operator not allowed: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _ALLOWED_BINOPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARYOPS:
            raise ValueError(f"Unary operator not allowed: {op_type.__name__}")
        return _ALLOWED_UNARYOPS[op_type](_safe_eval(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed")
        func_name = node.func.id
        if func_name not in _ALLOWED_FUNCS:
            raise ValueError(f"Function not allowed: {func_name}")
        args = [_safe_eval(a) for a in node.args]
        return _ALLOWED_FUNCS[func_name](*args)

    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTS:
            return _ALLOWED_CONSTS[node.id]
        raise ValueError(f"Name not allowed: {node.id}")

    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports +, -, *, /, **, %, //,
    and functions like sqrt(), log(), log10(), sin(), cos(), tan(), exp(),
    factorial(), and constants pi, e. Examples: 'sqrt(16)', 'log(100, 10)',
    'sin(pi/2)', '2**10 + sqrt(25)'."""
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _safe_eval(parsed)
        return f"Result: {result}"
    except ZeroDivisionError:
        return "Error calculating: division by zero"
    except Exception as e:
        return f"Error calculating: {e}"


# ==========================================================
# 5) UPLOADED PDF SUPPORT
# ==========================================================
uploaded_retriever = None
uploaded_filename = None


def process_uploaded_pdf(file_path: str, filename: str):
    """Load, chunk, embed, and store an uploaded PDF in an in-memory Chroma collection."""
    global uploaded_retriever, uploaded_filename

    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        if not docs:
            raise ValueError("No readable content found in the PDF.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embeddings = MistralAIEmbeddings(model="mistral-embed")
        temp_vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

        uploaded_retriever = temp_vectorstore.as_retriever(search_kwargs={"k": 4})
        uploaded_filename = filename
        return len(chunks)
    except Exception as e:
        uploaded_retriever = None
        uploaded_filename = None
        raise RuntimeError(f"Error processing PDF: {e}")


def clear_uploaded_pdf():
    global uploaded_retriever, uploaded_filename
    uploaded_retriever = None
    uploaded_filename = None


@tool
def search_uploaded_document(query: str) -> str:
    """Search the PDF that the user just uploaded in this session. Use this whenever
    the user refers to 'the file I uploaded', 'this document', or asks about content
    that isn't in the permanent internal knowledge base."""
    if uploaded_retriever is None:
        return "No document has been uploaded in this session yet."

    docs = uploaded_retriever.invoke(query)
    if not docs:
        return f"No relevant content found in the uploaded file ({uploaded_filename})."

    output = f"[Source: {uploaded_filename}]\n"
    for d in docs:
        output += f"\n{d.page_content}\n"
    return output