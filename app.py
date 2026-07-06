"""
Enterprise Agentic RAG System — Unified Production Application
--------------------------------------------------------------
Run with: streamlit run app.py
"""

import streamlit as st
import uuid
import tempfile
import os

# Import custom backend modules
from backend.agents.agent import agent, web_search
from backend.vision.image_handler import analyze_image_bytes
from backend.reports.report_generator import generate_report, save_report_to_file
from backend.retrievers.mmr import mmr_retriever
from backend.retrievers.multiquery import multiquery_retriever
from backend.retrievers.arxiv_retrivers import get_arxiv_retriever

# --- Same vectorstore used by mmr_retriever, so anything added here is
#     immediately searchable by chat and the report builder ---
from backend.vector_store.DB import vectorstore
from backend.agents.tools import process_uploaded_pdf, calculator
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Enterprise Workspace",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# MONOCHROME ENTERPRISE THEME (Sleek & Clean)
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stHeader"] {
    background-color: #0B0F19 !important;
    }

    [data-testid="stAppViewContainer"] {
       background-color: #0B0F19 !important;
    }

    [data-testid="stAppViewContainer"] > .main {
    background-color: #0B0F19 !important;
    }

    [data-testid="stBottomBlockContainer"] {
        background-color: #0B0F19 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1F2937;
    }
    .app-header {
        padding: 20px 0px 10px 0px;
        margin-bottom: 10px;
        text-align: center;
    }
    .app-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.5px;
    }
    .app-subtitle {
        font-size: 1.15rem;
        color: #9CA3AF;
        margin-top: 10px;
        max-width: 720px;
        margin-left: auto;
        margin-right: auto;
    }
    .tech-stack-line {
        font-size: 0.78rem;
        color: #6B7280;
        margin-top: 14px;
        letter-spacing: 0.2px;
    }
    .tech-stack-label {
        font-size: 0.7rem;
        color: #4B5563;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 12px;
        margin-bottom: 2px;
    }
    .footer-tech {
        text-align: center;
        margin-top: 40px;
        padding: 24px 0px;
        border-top: 1px solid #1F2937;
    }
    .footer-tech-label {
        font-size: 0.7rem;
        color: #4B5563;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }
    .footer-tech-list {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 6px 18px;
        font-size: 0.8rem;
        color: #9CA3AF;
    }
    .tech-badge {
        display: inline-block;
        background-color: #1A2033;
        border: 1px solid #374151;
        color: #A5B4FC;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        margin-right: 6px;
        margin-top: 14px;
    }
    div[data-testid="stChatInput"] {
        border-radius: 12px !important;
        background-color: #1F2937 !important;
    }
    .status-badge {
        background: linear-gradient(135deg, #16192b 0%, #111827 100%);
        border: 1px solid #2D3348;
        border-left: 3px solid #7F5AF0;
        padding: 18px 20px;
        border-radius: 12px;
        margin-bottom: 15px;
    }
    .status-badge-icon {
        font-size: 1.4rem;
        margin-bottom: 4px;
    }
    .output-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }
    .chip-btn button {
        border-radius: 999px !important;
        padding: 14px 32px !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        background: rgba(127, 90, 240, 0.12) !important;
        border: 1px solid rgba(127, 90, 240, 0.4) !important;
    }
    div.stButton > button {
        border-radius: 6px !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease;
    }
    button[kind="primary"] {
        background: linear-gradient(90deg, #7F5AF0, #6246EA) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 10px 0px !important;
        box-shadow: 0 4px 14px rgba(127, 90, 240, 0.35) !important;
    }
    .st-key-feature_card_pdf, .st-key-feature_card_web {
        border-radius: 18px !important;
        padding: 18px 20px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        transition: all 0.25s ease;
    }
    .st-key-feature_card_pdf {
        background: linear-gradient(135deg, rgba(127,90,240,0.18), rgba(127,90,240,0.03)) !important;
    }
    .st-key-feature_card_web {
        background: linear-gradient(135deg, rgba(56,132,255,0.18), rgba(56,132,255,0.03)) !important;
    }
    .st-key-feature_card_pdf:hover, .st-key-feature_card_web:hover {
        transform: translateY(-3px);
        border: 1px solid rgba(127, 90, 240, 0.4) !important;
    }
    .st-key-feature_card_web:hover {
        border: 1px solid rgba(56, 132, 255, 0.5) !important;
    }
    .feature-icon-badge {
        width: 46px; height: 46px;
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.3rem;
        margin-bottom: 10px;
    }
    .feature-icon-badge.purple { background: rgba(127,90,240,0.25); }
    .feature-icon-badge.blue { background: rgba(56,132,255,0.25); }
    .feature-card-title { font-size: 1.15rem; font-weight: 700; color: #fff; margin-bottom: 4px; margin-top: 4px; }
    .feature-card-desc { font-size: 0.85rem; color: #9CA3AF; line-height: 1.4; }
    .st-key-feature_card_pdf button, .st-key-feature_card_web button {
        border-radius: 999px !important;
        width: 40px !important; height: 40px !important;
        padding: 0px !important;
        font-size: 1.1rem !important;
        float: right;
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
    }
    .st-key-feature_card_pdf button:hover {
        background: rgba(127,90,240,0.35) !important;
        border: 1px solid rgba(127,90,240,0.6) !important;
    }
    .st-key-feature_card_web button:hover {
        background: rgba(56,132,255,0.35) !important;
        border: 1px solid rgba(56,132,255,0.6) !important;
    }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
   
</style>
<style>
/* Force all text white by default */
.stApp, .stApp p, .stApp span, .stApp label, .stApp div {
    color: #ffffff !important;
}

/* Sidebar text specifically */
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Expander header text */
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
    color: #ffffff !important;
}

/* File uploader text */
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] div {
    color: #ffffff !important;
}

/* Chat input placeholder + typed text */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea::placeholder {
    color: #ffffff !important;
}

/* Buttons text stays readable */
.stButton button {
    color: #ffffff !important;
}

/* Markdown headers/text in main area */
.stMarkdown, .stMarkdown p, .stMarkdown li {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SESSION STATE INITIALIZATION
# -----------------------------
defaults = {
    "messages": [],
    "report_count": 0,
    "docs_ingested": 0,
    "extracted_data": None,       # vision: raw extracted text
    "vision_report": None,        # vision: compiled report + path
    "vision_report_path": None,
    "topic_report": None,         # topic builder: compiled report + path
    "topic_report_path": None,
    "topic_report_title": None,
    "web_search_results": None,
    "web_search_query": None,
    "pdf_summary": None,          # pdf ingestion: quick summary
    "pdf_summary_name": None,
    "pdf_chunks_added": None,
    "pdf_page_count": None,
    "calc_result": None,
    "calc_expression": None,
}
for key, val in defaults.items():
    st.session_state.setdefault(key, val)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

SAMPLE_PROMPTS = [
    ("📄 Summarize Uploaded PDF", "Summarize the uploaded PDF into key points."),
    ("🌐 Latest AI Research Trends", "What are the latest AI research trends?"),
]

# -----------------------------
# SIDEBAR: CONTROLS ONLY (results render on main page)
# -----------------------------
with st.sidebar:
    st.markdown("### 🛠️ Workspace Controls")

    if st.button("➕ New Session", use_container_width=True, type="primary"):
        for key, val in defaults.items():
            st.session_state[key] = val
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.markdown("##### 📊 Document & Report Studio")

    # --- PDF upload -> adds directly to the same vectorstore used by mmr_retriever ---
    with st.expander("📄 Upload PDF", expanded=False):
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_ingest_uploader")

        if pdf_file and st.button("➕ Add to Knowledge Base", use_container_width=True):
            with st.spinner("Reading and indexing PDF..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(pdf_file.getvalue())
                        tmp_path = tmp.name

                    loader = PyPDFLoader(tmp_path)
                    pages = loader.load()

                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    chunks = splitter.split_documents(pages)

                    for chunk in chunks:
                        chunk.metadata["source"] = pdf_file.name

                    vectorstore.add_documents(chunks)
                    st.session_state.docs_ingested += 1
                    st.session_state.pdf_chunks_added = len(chunks)
                    st.session_state.pdf_page_count = len(pages)

                    # Also populate the session-scoped uploaded_retriever used by
                    # the search_uploaded_document agent tool
                    process_uploaded_pdf(tmp_path, pdf_file.name)

                    os.remove(tmp_path)

                    preview_text = "\n\n".join(c.page_content for c in chunks[:6])
                    quick_summary = generate_report(
                        f"Brief 4-5 sentence summary of '{pdf_file.name}'", [preview_text]
                    )
                    st.session_state.pdf_summary = quick_summary
                    st.session_state.pdf_summary_name = pdf_file.name
                    st.success("✅ Added to knowledge base — see summary on main page →")
                except Exception as e:
                    st.error(f"⚠️ Failed to ingest PDF: {e}")

        if st.session_state.docs_ingested:
            st.caption(f"📚 {st.session_state.docs_ingested} document(s) added this session")

    # --- Vision intelligence ---
    with st.expander("👁️ Vision Intelligence", expanded=False):
        uploaded_image = st.file_uploader("Upload Image/Chart", type=["png", "jpg", "jpeg"])
        if uploaded_image:
            if st.button("Extract Metrics", use_container_width=True):
                with st.spinner("Reading..."):
                    st.session_state.extracted_data = analyze_image_bytes(
                        image_bytes=uploaded_image.getvalue(),
                        mime_type=uploaded_image.type,
                        question="Extract all structured data."
                    )
                st.success("✅ Extracted — see result on main page →")

            if st.session_state.extracted_data:
                if st.button("Compile to Report", use_container_width=True):
                    with st.spinner("Compiling..."):
                        rep = generate_report("Vision Summary", [st.session_state.extracted_data])
                        path = save_report_to_file(rep, "vision_report.md")
                    st.session_state.report_count += 1
                    st.session_state.vision_report = rep
                    st.session_state.vision_report_path = path
                    st.success("✅ Report ready — see main page →")

    # --- Topic Report Builder ---
    with st.expander("📝 Topic Report Builder", expanded=False):
        topic = st.text_input("Target Topic", placeholder="e.g. Q3 Market Trends")
        c1, c2 = st.columns(2)
        use_arxiv = c1.checkbox("arXiv", value=True)
        use_mmr = c2.checkbox("Vector Store", value=True)

        if st.button("Generate Executive Brief", use_container_width=True):
            if topic:
                with st.spinner("Synthesizing..."):
                    src = []
                    if use_mmr: src += mmr_retriever.invoke(topic)
                    if use_arxiv: src.append(get_arxiv_retriever(topic, max_results=2))

                    rep = generate_report(topic, src)
                    path = save_report_to_file(rep, "brief.md")
                st.session_state.report_count += 1
                st.session_state.topic_report = rep
                st.session_state.topic_report_path = path
                st.session_state.topic_report_title = topic
                st.success("✅ Brief ready — see main page →")
            else:
                st.warning("Enter a topic")

    # --- Web Search ---
    with st.expander("🌐 Web Search", expanded=False):
        web_query = st.text_input("Search the web", placeholder="e.g. Latest Mistral AI model release")
        if st.button("Search", use_container_width=True, key="web_search_btn"):
            if web_query.strip():
                with st.spinner("Searching the web..."):
                    try:
                        results = web_search.invoke(web_query)
                        st.session_state.web_search_results = results
                        st.session_state.web_search_query = web_query
                        st.success("✅ Results ready — see main page →")
                    except Exception as e:
                        st.error(f"⚠️ Web search failed: {e}")
            else:
                st.warning("Enter a search query")

    # --- Calculator ---
    with st.expander("🧮 Calculator", expanded=False):
        calc_expression = st.text_input(
            "Expression", placeholder="e.g. (245 * 12) / 4 + 18",
            key="calc_expression_input"
        )
        if st.button("Calculate", use_container_width=True, key="calc_btn"):
            if calc_expression.strip():
                result = calculator.invoke(calc_expression)
                st.session_state.calc_result = result
                st.session_state.calc_expression = calc_expression
            else:
                st.warning("Enter an expression")

        if st.session_state.get("calc_result"):
            st.markdown(
                f"<div style='background-color:#1F2937; border:1px solid #374151; "
                f"border-radius:8px; padding:10px 14px; margin-top:8px; font-family:monospace; "
                f"color:#A5B4FC; font-size:0.9rem;'>{st.session_state.calc_expression} → "
                f"<b>{st.session_state.calc_result}</b></div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("##### 🟢 System Telemetry")
    st.markdown(f"""
    <div style='font-size:0.82rem; line-height:2.1;'>
        <div style='display:flex; justify-content:space-between;'><span>🤖 AI Agent</span><span style='color:#4ADE80;'>Active</span></div>
        <div style='display:flex; justify-content:space-between;'><span>🧠 LLM</span><span style='color:#9CA3AF;'>Mistral AI</span></div>
        <div style='display:flex; justify-content:space-between;'><span>🗄️ Vector DB</span><span style='color:#4ADE80;'>Connected</span></div>
        <div style='display:flex; justify-content:space-between;'><span>📄 Documents</span><span style='color:#9CA3AF;'>{st.session_state.docs_ingested}</span></div>
        <div style='display:flex; justify-content:space-between;'><span>📊 Reports</span><span style='color:#9CA3AF;'>{st.session_state.report_count}</span></div>
        <div style='display:flex; justify-content:space-between;'><span>🔍 Retrieval</span><span style='color:#9CA3AF;'>MMR + Multi Query</span></div>
        <div style='display:flex; justify-content:space-between;'><span>🌐 Web Search</span><span style='color:#4ADE80;'>Connected</span></div>
        <div style='display:flex; justify-content:space-between;'><span>👁️ Vision AI</span><span style='color:#4ADE80;'>Ready</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Clear Data", use_container_width=True):
        clear_keys = [
            "extracted_data", "vision_report", "vision_report_path",
            "topic_report", "topic_report_path", "topic_report_title",
            "web_search_results", "web_search_query",
            "pdf_summary", "pdf_summary_name", "pdf_chunks_added", "pdf_page_count",
            "calc_result", "calc_expression",
        ]
        for key in clear_keys:
            st.session_state[key] = defaults[key]
        st.success("✅ Cleared all workspace data")
        st.rerun()

    st.divider()
    st.markdown(
        "<div style='text-align:center; font-size:0.8rem; color:#D1D5DB; font-weight:600;'>"
        "Enterprise Agentic RAG System"
        "</div>"
        "<div style='text-align:center; font-size:0.7rem; color:#6B7280; margin-bottom:10px;'>"
        "Version 1.0"
        "</div>"
        "<div style='text-align:center; font-size:0.68rem; color:#4B5563; text-transform:uppercase; letter-spacing:1px;'>"
        "Built with"
        "</div>"
        "<div style='text-align:center; font-size:0.75rem; color:#9CA3AF;'>"
        "LangChain • LangGraph"
        "</div>",
        unsafe_allow_html=True,
    )

# -----------------------------
# MAIN WORKSPACE
# -----------------------------
st.markdown("""
<div class="app-header">
    <div class="app-title">⚡ Enterprise Agentic RAG System</div>
    
</div>
""", unsafe_allow_html=True)

# -----------------------------
# WORKSPACE OUTPUT — results from sidebar actions render here
# -----------------------------
has_output = any([
    st.session_state.pdf_summary,
    st.session_state.extracted_data,
    st.session_state.vision_report,
    st.session_state.topic_report,
    st.session_state.web_search_results,
])

if has_output:
    col_h1, col_h2 = st.columns([5, 1])
    with col_h1:
        st.markdown('<div class="section-header" style="color:#fff; font-size:1.1rem; font-weight:600; border-left:3px solid #7F5AF0; padding-left:10px;">📊 Workspace Output</div>', unsafe_allow_html=True)
    with col_h2:
        if st.button("🔄 Reset", use_container_width=True, key="reset_output_btn"):
            st.session_state.extracted_data = None
            st.session_state.vision_report = None
            st.session_state.vision_report_path = None
            st.session_state.topic_report = None
            st.session_state.topic_report_path = None
            st.session_state.topic_report_title = None
            st.session_state.pdf_summary = None
            st.session_state.pdf_summary_name = None
            st.session_state.pdf_chunks_added = None
            st.session_state.web_search_results = None
            st.session_state.web_search_query = None
            st.rerun()

    if st.session_state.pdf_summary:
        with st.container():
            st.markdown('<div class="output-card">', unsafe_allow_html=True)
            c1, c2 = st.columns([6, 1])
            c1.markdown(f"**📄 PDF Summary — `{st.session_state.pdf_summary_name}`**")
            if c2.button("✖", key="clear_pdf_summary"):
                st.session_state.pdf_summary = None
                st.session_state.pdf_summary_name = None
                st.session_state.pdf_chunks_added = None
                st.session_state.pdf_page_count = None
                st.rerun()
            st.caption(f"{st.session_state.pdf_page_count} page(s) · {st.session_state.pdf_chunks_added} chunks added to knowledge base")
            st.markdown(st.session_state.pdf_summary)
            st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.extracted_data:
        with st.container():
            st.markdown('<div class="output-card">', unsafe_allow_html=True)
            c1, c2 = st.columns([6, 1])
            c1.markdown("**📷 Extracted Data (Vision)**")
            if c2.button("✖", key="clear_extracted_data"):
                st.session_state.extracted_data = None
                st.rerun()
            st.code(st.session_state.extracted_data, language="text")
            st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.vision_report:
        with st.container():
            st.markdown('<div class="output-card">', unsafe_allow_html=True)
            c1, c2 = st.columns([6, 1])
            c1.markdown("**📊 Vision Report**")
            if c2.button("✖", key="clear_vision_report"):
                st.session_state.vision_report = None
                st.session_state.vision_report_path = None
                st.rerun()
            st.markdown(st.session_state.vision_report)
            st.caption(f"📁 Saved to: `{st.session_state.vision_report_path}`")
            with open(st.session_state.vision_report_path, "rb") as f:
                st.download_button(
                    "⬇️ Download to PC", data=f, file_name="vision_report.md",
                    mime="text/markdown", key="download_vision_report",
                )
            st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.topic_report:
        with st.container():
            st.markdown('<div class="output-card">', unsafe_allow_html=True)
            c1, c2 = st.columns([6, 1])
            c1.markdown(f"**📝 Executive Brief — {st.session_state.topic_report_title}**")
            if c2.button("✖", key="clear_topic_report"):
                st.session_state.topic_report = None
                st.session_state.topic_report_path = None
                st.session_state.topic_report_title = None
                st.rerun()
            st.markdown(st.session_state.topic_report)
            st.caption(f"📁 Saved to: `{st.session_state.topic_report_path}`")
            with open(st.session_state.topic_report_path, "rb") as f:
                st.download_button(
                    "⬇️ Download to PC", data=f, file_name="brief.md",
                    mime="text/markdown", key="download_brief_report",
                )
            st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.web_search_results:
        with st.container():
            st.markdown('<div class="output-card">', unsafe_allow_html=True)
            c1, c2 = st.columns([6, 1])
            c1.markdown(f"**🌐 Web Search — \"{st.session_state.web_search_query}\"**")
            if c2.button("✖", key="clear_web_search"):
                st.session_state.web_search_results = None
                st.session_state.web_search_query = None
                st.rerun()
            st.markdown(st.session_state.web_search_results)
            st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

# -----------------------------
# CONVERSATIONAL WORKSPACE
# -----------------------------

# Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# SELF-HEALING: process any pending (unanswered) user query
# -----------------------------
# If a run gets interrupted (e.g. the user sends another message or clicks
# something else while the agent is still "thinking"), Streamlit kills the
# in-progress script and the previous user message can end up in
# st.session_state.messages with no assistant reply. That produces answers
# out of order in the chat. To guarantee every user message is always
# immediately followed by its own answer, we never call the agent directly
# inside a button/chat_input handler — we only append the user message and
# rerun. This block (which always runs first, right after the history) is
# the single place that actually calls the agent and appends the reply.
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    pending_query = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant"):
        with st.spinner("Processing Agent Pipelines..."):
            try:
                result = agent.invoke(
                    {"messages": [("user", pending_query)]},
                    config={"configurable": {"thread_id": st.session_state.thread_id}},
                )
                response = result["messages"][-1].content
            except Exception as e:
                response = f"⚠️ Pipeline Error: {e}"
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Hero Section (Only before first message) — shows the two option cards.
# As soon as the user picks one, a message gets appended to st.session_state.messages
# and st.rerun() fires, so this whole block naturally disappears on the next render.
if len(st.session_state.messages) == 0:

    st.markdown("""
<div style="text-align:center; margin:20px 0 30px 0;">
    <h3 style="color:white; margin-bottom:10px; font-size:1.5rem; font-weight:700; letter-spacing:-0.2px;">
         Your Enterprise AI Workspace
    </h3>
    <div style="color:#9CA3AF; font-size:0.9rem; font-weight:500; letter-spacing:0.3px;">
        🔎 Research &nbsp;•&nbsp; 📄 Documents &nbsp;•&nbsp; 👁️ Vision &nbsp;•&nbsp; 🌐 Web Search
    </div>
</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # -------------------- CARD 1: SUMMARIZE PDF (Purple theme) --------------------
    with col1:
        with st.container(key="feature_card_pdf"):
            btn_col1, btn_col2 = st.columns([5, 1])
            with btn_col1:
                st.markdown('<div class="feature-icon-badge purple">📄</div>', unsafe_allow_html=True)
            with btn_col2:
                clicked_pdf_card = st.button("→", key="sug_0", help="Summarize Uploaded PDF")

            st.markdown('<div class="feature-card-title">Summarize Uploaded PDF</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="feature-card-desc">Generate concise summaries and key insights from your documents.</div>',
                unsafe_allow_html=True,
            )

        if clicked_pdf_card:
            query = "Summarize the uploaded PDF into key points."
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()

    # -------------------- CARD 2: LATEST AI RESEARCH TRENDS (Blue theme) --------------------
    with col2:
        with st.container(key="feature_card_web"):
            btn_col1, btn_col2 = st.columns([5, 1])
            with btn_col1:
                st.markdown('<div class="feature-icon-badge blue">🌐</div>', unsafe_allow_html=True)
            with btn_col2:
                clicked_web_card = st.button("→", key="sug_1", help="Latest AI Research Trends")

            st.markdown('<div class="feature-card-title">Latest AI Research Trends</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="feature-card-desc">Explore the latest AI advancements and industry trends.</div>',
                unsafe_allow_html=True,
            )

        if clicked_web_card:
            query = "What are the latest AI research trends?"
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()

# -----------------------------
# CHAT INPUT — always available at the bottom
# -----------------------------
chat_submission = st.chat_input(
    "Query anything over your knowledge ecosystem...",
    accept_file=True,
    file_type=["pdf"],
)

if chat_submission:
    chat_query = chat_submission.text
    attached_files = chat_submission.files or []

    # If a PDF was attached via the "+" icon, ingest it the same way as the sidebar uploader
    for f in attached_files:
        with st.spinner(f"Indexing '{f.name}'..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.getvalue())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = splitter.split_documents(pages)
                for chunk in chunks:
                    chunk.metadata["source"] = f.name

                vectorstore.add_documents(chunks)
                process_uploaded_pdf(tmp_path, f.name)
                st.session_state.docs_ingested += 1
                st.session_state.pdf_chunks_added = len(chunks)
                st.session_state.pdf_page_count = len(pages)
                os.remove(tmp_path)

                # Generate a quick summary so it shows up persistently in Workspace Output
                preview_text = "\n\n".join(c.page_content for c in chunks[:6])
                quick_summary = generate_report(
                    f"Brief 4-5 sentence summary of '{f.name}'", [preview_text]
                )
                st.session_state.pdf_summary = quick_summary
                st.session_state.pdf_summary_name = f.name

                st.toast(f"✅ '{f.name}' added to knowledge base ({len(pages)} pages)")
            except Exception as e:
                st.error(f"⚠️ Failed to ingest '{f.name}': {e}")

    if chat_query:
        st.session_state.messages.append({"role": "user", "content": chat_query})

    st.rerun()