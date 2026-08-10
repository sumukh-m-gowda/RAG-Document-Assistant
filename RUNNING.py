"""
RAG Document Assistant — Streamlit App (Day 6 + Day 7, minimized)

Pipeline: upload PDFs -> chunk -> embed -> Chroma -> condense question ->
retrieve (filtered) -> answer from docs, or fall back to general knowledge ->
optionally call the email tool if the user asked for one (hybrid agent).

Run with:  streamlit run app.py
"""

import os
import re
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY2 = os.getenv("GEMINI_API_KEY2")

DATA_DIR = "data"
PERSIST_DIR = "chroma_db"
# Every rebuild gets its own Chroma collection name instead of deleting the old one —
# avoids Windows file-lock errors from a previous client still holding the folder open.
ACTIVE_COLLECTION_FILE = os.path.join(PERSIST_DIR, "active_collection.txt")
DEFAULT_THRESHOLD = 0.8
EMAIL_KEYWORDS = ("email", "mail", "send")

st.set_page_config(page_title="RAG Document Assistant", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def get_embeddings_model():
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", google_api_key=GEMINI_API_KEY2)


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=GEMINI_API_KEY2)


@st.cache_resource(show_spinner=False)
def get_llm_with_tools():
    """Same model as get_llm(), with the email tool bound (Day 1's bind_tools pattern)."""
    return get_llm().bind_tools([email_this])


# ---------------------------------------------------------------------------
# Collection tracking (which Chroma collection is "current")
# ---------------------------------------------------------------------------
def _read_active_collection():
    if os.path.isfile(ACTIVE_COLLECTION_FILE):
        with open(ACTIVE_COLLECTION_FILE, "r") as f:
            return f.read().strip()
    return None


def _write_active_collection(name):
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(ACTIVE_COLLECTION_FILE, "w") as f:
        f.write(name)


def load_existing_store():
    """Load the most recently built Chroma collection from disk, if one exists."""
    collection_name = _read_active_collection()
    if not (collection_name and os.path.isdir(PERSIST_DIR)):
        return None
    store = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings_model(),
        collection_name=collection_name,
    )
    return store if store._collection.count() > 0 else None


# ---------------------------------------------------------------------------
# Rate-limit-safe batched embedding
# ---------------------------------------------------------------------------
def _extract_retry_delay(error_message: str):
    """Reuse Google's own 'Please retry in 43.4s' hint from a 429 error, if present."""
    match = re.search(r"retry in ([\d.]+)s", error_message, re.IGNORECASE)
    return float(match.group(1)) + 1 if match else None


def _add_batch_with_retry(store, batch, progress_callback=None, max_retries=2):
    """Retry a batch on 429 rate-limit errors (capped wait, fails fast otherwise)."""
    for attempt in range(max_retries):
        try:
            store.add_documents(batch)
            return
        except Exception as e:
            msg = str(e)
            is_rate_limit = "RESOURCE_EXHAUSTED" in msg or "429" in msg
            if not is_rate_limit or attempt == max_retries - 1:
                raise
            wait = min(_extract_retry_delay(msg) or 30, 65)
            if progress_callback:
                progress_callback(f"Rate limit hit — waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
    raise RuntimeError("Failed to embed after retries due to API rate limits.")


def build_vector_store_batched(chunks, collection_name, batch_size=50, pause_seconds=2, progress_callback=None):
    """Embed in batches (fewer, larger requests) with a pause between them."""
    store = Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embeddings_model(), collection_name=collection_name)
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        if progress_callback:
            progress_callback(f"Embedding chunks {i + 1}-{min(i + batch_size, total)} of {total}...")
        _add_batch_with_retry(store, batch, progress_callback=progress_callback)
        if i + batch_size < total:
            time.sleep(pause_seconds)
    return store


def build_store_from_uploads(uploaded_files, progress_callback=None):
    """Save uploads -> load -> split -> embed (batched) -> persist a fresh collection.
    Only processes files from THIS call, so old data/ leftovers don't get re-embedded."""
    os.makedirs(DATA_DIR, exist_ok=True)
    saved_paths = []
    for uf in uploaded_files:
        path = os.path.join(DATA_DIR, uf.name)
        with open(path, "wb") as f:
            f.write(uf.getbuffer())
        saved_paths.append(path)

    documents = []
    for path in saved_paths:
        documents.extend(PyPDFLoader(path).load())
    if not documents:
        return None, 0

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
    collection_name = f"docs_{int(time.time())}"
    store = build_vector_store_batched(chunks, collection_name, progress_callback=progress_callback)
    _write_active_collection(collection_name)
    return store, len(chunks)


# ---------------------------------------------------------------------------
# Email (SMTP) — plain function + LLM-callable tool wrapper
# ---------------------------------------------------------------------------
def send_email(recipient_email: str, subject: str, body: str) -> str:
    """Send an email via Gmail SMTP. Returns a success or error message (never raises)."""
    sender_email = os.environ.get('SENDER_EMAIL')
    app_password = os.environ.get('SENDER_APP_PASSWORD')
    try:
        msg = MIMEMultipart()
        msg['From'], msg['To'], msg['Subject'] = sender_email, recipient_email, subject
        msg.attach(MIMEText(body, 'plain'))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        return f'Email successfully sent to {recipient_email} with subject: {subject}'
    except smtplib.SMTPAuthenticationError:
        return 'Authentication failed. Make sure you are using a Gmail App Password, not your regular password.'
    except smtplib.SMTPRecipientsRefused:
        return f'Recipient {recipient_email} was refused. Check if the email address is correct.'
    except Exception as e:
        return f'Failed to send email: {str(e)}'


@tool
def email_this(recipient_email: str, subject: str, message: str) -> str:
    """Send an email to the given recipient with the given subject and message.
    Use ONLY when the user explicitly asks to email/send/mail something to an address."""
    return send_email(recipient_email, subject, message)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
CONDENSE_PROMPT = ChatPromptTemplate.from_template(
    """Given the conversation history and a follow-up question, rewrite the follow-up
question to be a standalone question that includes all necessary context from the
history. If already standalone, return it unchanged. Do not answer it — only rewrite it.

Conversation history:
{chat_history}

Follow-up question: {question}

Standalone question:"""
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant having an ongoing conversation, answering using ONLY the context below.

Rules:
- Base your answer strictly on the context. Do not use outside knowledge.
- Reference the relevant [Source N] tag(s) inline when used.
- Be concise and direct.

Context:
{context}

Question:
{question}

Answer:"""
)

FALLBACK_PROMPT = ChatPromptTemplate.from_template(
    """The user's question isn't covered by their uploaded documents.
Answer it using your own general knowledge, and be clear that this isn't from their documents.

Question:
{question}

Answer:"""
)

HYBRID_INSTRUCTIONS = """You are a helpful assistant.

Rules:
- If the uploaded documents contain the answer, answer using ONLY the documents and cite [Source N].
- If the documents do NOT contain the answer, answer using your own general knowledge.
- The user has asked you to email something — use the email_this tool to do so.
- Never call the email tool unless the user explicitly asked for an email.

Context:
{context}

Question:
{question}
"""


# ---------------------------------------------------------------------------
# Core RAG logic
# ---------------------------------------------------------------------------
def format_history(chat_history):
    if not chat_history:
        return "(no previous conversation)"
    return "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in chat_history
    )


def condense_question(question, chat_history, llm):
    if not chat_history:
        return question
    chain = CONDENSE_PROMPT | llm | StrOutputParser()
    return chain.invoke({"chat_history": format_history(chat_history), "question": question}).strip()


def retrieve_filtered(store, query, k=5, threshold=DEFAULT_THRESHOLD):
    results = store.similarity_search_with_score(query, k=k)
    return [(doc, score) for doc, score in results if score <= threshold]


def format_docs_with_citations(filtered_results):
    if not filtered_results:
        return "NO_RELEVANT_CONTEXT", {}
    context_parts, citation_map = [], {}
    for i, (doc, score) in enumerate(filtered_results, 1):
        tag = f"Source {i}"
        citation_map[tag] = f"{os.path.basename(doc.metadata.get('source', 'unknown'))} (page {doc.metadata.get('page', '?')})"
        context_parts.append(f"[{tag}]\n{doc.page_content}")
    return "\n\n".join(context_parts), citation_map


def generate_grounded_answer(llm, context, question):
    """Answer from the documents, or fall back to general knowledge if nothing relevant
    was retrieved. This is the one place both agent_mode branches reuse — previously
    duplicated in three places."""
    if context == "NO_RELEVANT_CONTEXT":
        chain = FALLBACK_PROMPT | llm | StrOutputParser()
        return chain.invoke({"question": question})
    chain = ANSWER_PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


def extract_text(content):
    """Gemini tool-calling responses sometimes return content as a list of parts
    instead of a plain string — normalize either shape into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(item.get("text", "") for item in content if isinstance(item, dict))
    return str(content)


def answer_question(store, llm, question, chat_history, threshold, agent_mode=False):
    standalone = condense_question(question, chat_history, llm)
    filtered = retrieve_filtered(store, standalone, threshold=threshold)
    context, citation_map = format_docs_with_citations(filtered)

    wants_email = agent_mode and any(w in question.lower() for w in EMAIL_KEYWORDS)

    if not wants_email:
        answer = generate_grounded_answer(llm, context, standalone)
        return answer, citation_map, standalone

    # --- Hybrid agent path: same shape as Day 1's run_agent() ---
    # send messages -> check tool_calls -> if present, execute + feed result back -> final answer.
    llm_tools = get_llm_with_tools()
    messages = [HumanMessage(content=HYBRID_INSTRUCTIONS.format(context=context, question=standalone))]
    response = llm_tools.invoke(messages)
    messages.append(response)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            messages.append(email_this.invoke(tool_call))
        answer = extract_text(llm_tools.invoke(messages).content)
    else:
        answer = extract_text(response.content)

    return answer, citation_map, standalone


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "store" not in st.session_state:
    st.session_state.store = load_existing_store()

with st.sidebar:
    st.header("📄 Knowledge Base")

    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY not found. Add it to your .env file.")

    uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    notify_email = st.text_input(
        "Notify this email on upload",
        value=os.environ.get("SENDER_EMAIL", ""),
        help="Sends a confirmation email via Gmail SMTP once the knowledge base is built."
    )

    if st.button("Build / Rebuild Knowledge Base", disabled=not uploaded_files):
        status_box = st.empty()
        show_progress = lambda msg: status_box.info(msg)

        store, n_chunks, build_failed = None, 0, False
        try:
            store, n_chunks = build_store_from_uploads(uploaded_files, progress_callback=show_progress)
        except Exception as e:
            build_failed = True
            status_box.empty()
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                st.error("Gemini's free-tier embedding quota (100/min) was exceeded even after retrying. Wait a minute and try again, or upload fewer/smaller files.")
            else:
                st.error(f"Failed to build the knowledge base: {e}")
        status_box.empty()

        if build_failed:
            pass
        elif store is None:
            st.error("No readable text found in the uploaded file(s).")
        else:
            st.session_state.store = store
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.success(f"Indexed {n_chunks} chunk(s) from {len(uploaded_files)} file(s).")

            if notify_email:
                body = (
                    f"Your RAG Document Assistant just finished indexing new documents.\n\n"
                    f"Files uploaded: {', '.join(f.name for f in uploaded_files)}\n"
                    f"Chunks indexed: {n_chunks}\n\n"
                    f"You can now ask questions about these documents in the app."
                )
                with st.spinner(f"Sending notification email to {notify_email}..."):
                    result = send_email(notify_email, "Your RAG Knowledge Base Was Updated", body)
                (st.success if result.startswith("Email successfully sent") else st.warning)(result)

    st.divider()
    threshold = st.slider(
        "Similarity threshold (lower = stricter)",
        min_value=0.3, max_value=1.5, value=DEFAULT_THRESHOLD, step=0.05,
        help="Chunks scoring above this distance are treated as not relevant."
    )

    st.divider()
    if st.button("🗑️ New Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    agent_mode = st.checkbox(
        "🤖 Enable email tool (hybrid agent)",
        value=False,
        help='Lets the model send an email when you explicitly ask, e.g. "email this summary to alex@gmail.com".'
    )

    if st.session_state.store is not None:
        st.caption(f"Knowledge base: {st.session_state.store._collection.count()} chunks indexed")

st.title("📄 RAG Document Assistant")
st.caption("Ask questions grounded in your own documents. Answers cite the source chunk they came from.")

if st.session_state.store is None:
    st.info("Upload one or more PDFs in the sidebar and click **Build / Rebuild Knowledge Base** to get started.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for tag, label in msg["sources"].items():
                        st.markdown(f"**{tag}** — {label}")

    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input, "sources": None})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, citation_map, standalone = answer_question(
                    st.session_state.store, get_llm(), user_input,
                    st.session_state.chat_history, threshold, agent_mode
                )
                st.markdown(answer)
                if citation_map:
                    with st.expander("Sources"):
                        for tag, label in citation_map.items():
                            st.markdown(f"**{tag}** — {label}")
                if standalone != user_input:
                    st.caption(f'(interpreted as: "{standalone}")')

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": citation_map})
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=answer))