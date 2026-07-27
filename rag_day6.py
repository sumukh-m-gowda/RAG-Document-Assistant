"""
Run with:  streamlit run rag_day6.py

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

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DATA_DIR = "data"
PERSIST_DIR = "chroma_db"
# Chroma keeps its persisted files open while a client object is alive. On Windows,
# trying to shutil.rmtree() that folder while a previous Chroma client still has it
# open raises PermissionError: [WinError 32]. Rather than deleting anything, we give
# every rebuild its own collection name (a "table" inside the same chroma_db folder)
# and remember which one is current — no file deletion required.
ACTIVE_COLLECTION_FILE = os.path.join(PERSIST_DIR, "active_collection.txt")
DEFAULT_THRESHOLD = 0.8

st.set_page_config(page_title="RAG Document Assistant", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def get_embeddings_model():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY)


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)


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
    if collection_name and os.path.isdir(PERSIST_DIR):
        store = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=get_embeddings_model(),
            collection_name=collection_name,
        )
        if store._collection.count() > 0:
            return store
    return None


def _extract_retry_delay(error_message: str):
    """Google's 429 errors often include 'Please retry in 43.4s' — reuse that hint if present."""
    match = re.search(r"retry in ([\d.]+)s", error_message, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1  # small buffer on top of the suggested delay
    return None


def _add_batch_with_retry(store, batch, progress_callback=None, max_retries=2):
    """Add one batch of chunks to the store, retrying with backoff on rate-limit (429) errors.
    Capped at 2 retries with a max ~65s wait each — fails fast instead of silently blocking
    for many minutes if the free-tier quota is still cooling down."""
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
                progress_callback(f"Rate limit hit — waiting {wait:.0f}s before retrying (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
    raise RuntimeError("Failed to embed after retries due to API rate limits.")


def build_vector_store_batched(chunks, collection_name, batch_size=50, pause_seconds=2, progress_callback=None):
    """
    Embed chunks in batches, pausing briefly between them. Using fewer, larger batches
    (default 50 chunks/request) means small documents finish in a single request instead
    of being split into many small ones that each risk tripping the per-minute rate limit.
    """
    store = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings_model(),
        collection_name=collection_name,
    )
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
    """Save uploaded PDFs to disk, load + split + embed (batched) + persist a fresh Chroma collection.
    Only processes the files uploaded in THIS call — old files left over in data/ from
    previous sessions are ignored, so chunk counts stay accurate to what you just uploaded."""
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

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Every rebuild gets its own collection name instead of deleting the old one —
    # this sidesteps Windows file-lock errors entirely (see comment above load_existing_store).
    collection_name = f"docs_{int(time.time())}"

    store = build_vector_store_batched(chunks, collection_name, progress_callback=progress_callback)
    _write_active_collection(collection_name)
    return store, len(chunks)


# ---------------------------------------------------------------------------
# Email notification (reused from the Day 1 email agent project)
# ---------------------------------------------------------------------------
def send_email(recipient_email: str, subject: str, body: str) -> str:
    """
    Send an email to a given recipient email address with a given subject and body.
    Uses Gmail SMTP to deliver the email. Returns success or error message.
    """
    sender_email = os.environ.get('SENDER_EMAIL')
    app_password = os.environ.get('SENDER_APP_PASSWORD')
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
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


# ---------------------------------------------------------------------------
# RAG logic (Days 2-5, unchanged in behavior)
# ---------------------------------------------------------------------------
CONDENSE_PROMPT = ChatPromptTemplate.from_template(
    """Given the conversation history and a follow-up question, rewrite the follow-up
question to be a standalone question that includes all necessary context from the
history. If the follow-up question is already standalone, return it unchanged.
Do not answer the question - only rewrite it.

Conversation history:
{chat_history}

Follow-up question: {question}

Standalone question:"""
)

FALLBACK_PROMPT = ChatPromptTemplate.from_template(
    """The user's question isn't covered by their uploaded documents. Answer it using
your own general knowledge instead, and start your answer with:
"(Not from your documents — general knowledge:)"

Question:
{question}

Answer:"""
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant having an ongoing conversation, answering questions
using ONLY the context provided below.

Rules:
- Base your answer strictly on the context. Do not use outside knowledge.
- Reference the relevant [Source N] tag(s) inline when you use information from them.
- If the context is exactly "NO_RELEVANT_CONTEXT", respond only with:
  "I don't have relevant information in the provided documents to answer that."
- You may refer naturally to earlier parts of the conversation, but never invent
  facts that aren't in the context.
- Be concise and direct.

Context:
{context}

Question:
{question}

Answer:"""
)


def format_history(chat_history):
    if not chat_history:
        return "(no previous conversation)"
    lines = []
    for msg in chat_history:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


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
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        citation_map[tag] = f"{os.path.basename(source)} (page {page})"
        context_parts.append(f"[{tag}]\n{doc.page_content}")
    return "\n\n".join(context_parts), citation_map


def answer_question(store, llm, question, chat_history, threshold):
    standalone = condense_question(question, chat_history, llm)
    filtered = retrieve_filtered(store, standalone, threshold=threshold)
    context, citation_map = format_docs_with_citations(filtered)
    if context == "NO_RELEVANT_CONTEXT" and allow_fallback:
        chain = FALLBACK_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"question": standalone})
    else:
        chain = ANSWER_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": standalone})
    return answer, citation_map, standalone


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: {role, content, sources}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of HumanMessage/AIMessage for condensing
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

        def show_progress(msg):
            status_box.info(msg)

        store, n_chunks, build_failed = None, 0, False
        try:
            store, n_chunks = build_store_from_uploads(uploaded_files, progress_callback=show_progress)
        except Exception as e:
            build_failed = True
            status_box.empty()
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                st.error(
                    "Gemini's free-tier embedding quota (100 requests/minute) was exceeded even "
                    "after retrying. Wait a minute and try again, or upload fewer/smaller files."
                )
            else:
                st.error(f"Failed to build the knowledge base: {e}")

        status_box.empty()

        if build_failed:
            pass  # error already shown above
        elif store is None:
            st.error("No readable text found in the uploaded file(s).")
        else:
            st.session_state.store = store
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.success(f"Indexed {n_chunks} chunk(s) from {len(uploaded_files)} file(s).")

            if notify_email:
                file_names = ", ".join(f.name for f in uploaded_files)
                subject = "Your RAG Knowledge Base Was Updated"
                body = (
                    f"Your RAG Document Assistant just finished indexing new documents.\n\n"
                    f"Files uploaded: {file_names}\n"
                    f"Chunks indexed: {n_chunks}\n\n"
                    f"You can now ask questions about these documents in the app."
                )
                with st.spinner(f"Sending notification email to {notify_email}..."):
                    email_result = send_email(notify_email, subject, body)
                if email_result.startswith("Email successfully sent"):
                    st.success(email_result)
                else:
                    st.warning(email_result)

    st.divider()

    threshold = st.slider(
        "Similarity threshold (lower = stricter)",
        min_value=0.3, max_value=1.5, value=DEFAULT_THRESHOLD, step=0.05,
        help="Chunks scoring above this distance are treated as not relevant."
    )

    allow_fallback = st.checkbox(
        "Allow general knowledge when documents don't cover it",
        value=False,
        help="If off, the assistant only answers from your uploaded documents."
    )
    
    st.divider()
    if st.button("🗑️ New Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

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
                llm = get_llm()
                answer, citation_map, standalone = answer_question(
                    st.session_state.store, llm, user_input,
                    st.session_state.chat_history, threshold
                )
                st.markdown(answer)
                if citation_map:
                    with st.expander("Sources"):
                        for tag, label in citation_map.items():
                            st.markdown(f"**{tag}** — {label}")
                if standalone != user_input:
                    st.caption(f"(interpreted as: \"{standalone}\")")

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": citation_map})
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=answer))