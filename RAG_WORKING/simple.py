"""
RAG Document Assistant — SIMPLE version

Same features as the full app (upload PDFs, chat, citations, memory, optional
email tool) but written as plainly as possible — no retry loops, no batching,
no keyword pre-checks. This version is for UNDERSTANDING the flow.
See app.py for the hardened version you'd actually deploy.

Run with:  streamlit run app_simple.py
"""

import os
import shutil
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

# ===========================================================================
# STEP 1 — Setup
# ===========================================================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DATA_DIR = "data"
PERSIST_DIR = "chroma_db"

st.set_page_config(page_title="RAG Document Assistant (Simple)", page_icon="📄")


@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)


@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)


# ===========================================================================
# STEP 2 — Build the knowledge base (Days 2 & 6, simplified)
# ===========================================================================
def build_knowledge_base(uploaded_files):
    """Save uploaded PDFs -> load -> split -> embed -> return a Chroma store.
    Simple version: no batching, no retry, one big embed call. If you hit a
    rate-limit error here on a large PDF, just wait a minute and try again."""

    # Save the uploaded files to disk
    os.makedirs(DATA_DIR, exist_ok=True)
    documents = []
    for uploaded_file in uploaded_files:
        path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        documents.extend(PyPDFLoader(path).load())

    if not documents:
        return None

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Start fresh each time — delete the old database folder if it exists.
    # (Simple approach: if this fails because a file is still locked, just
    # restart the app once — this only happens right after a rebuild.)
    if os.path.isdir(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)

    # Embed everything and save to disk, in one call
    store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=PERSIST_DIR,
    )
    return store


def load_knowledge_base_if_exists():
    """On app startup, reload a previously built database if one is on disk."""
    if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embeddings())
    return None


# ===========================================================================
# STEP 3 — Email sending (Day 1's function, unchanged)
# ===========================================================================
def send_email(recipient_email: str, subject: str, body: str) -> str:
    """Send an email via Gmail SMTP. Returns a success or error message."""
    sender_email = os.environ.get("SENDER_EMAIL")
    app_password = os.environ.get("SENDER_APP_PASSWORD")
    try:
        msg = MIMEMultipart()
        msg["From"], msg["To"], msg["Subject"] = sender_email, recipient_email, subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        return f"Email sent to {recipient_email}."
    except Exception as e:
        return f"Failed to send email: {e}"


@tool
def email_this(recipient_email: str, subject: str, message: str) -> str:
    """Send an email. Use ONLY when the user explicitly asks to email/send something."""
    return send_email(recipient_email, subject, message)


# ===========================================================================
# STEP 4 — Prompts (Days 3, 4, 5, 7)
# ===========================================================================
CONDENSE_PROMPT = ChatPromptTemplate.from_template(
    """Rewrite the follow-up question as a standalone question using the chat history.
If it's already standalone, return it unchanged. Only rewrite it — don't answer it.

Chat history:
{chat_history}

Follow-up question: {question}

Standalone question:"""
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below. Cite [Source N] when you use it.
If the context doesn't contain the answer, say so clearly instead of guessing.

Context:
{context}

Question:
{question}

Answer:"""
)


# ===========================================================================
# STEP 5 — The RAG pipeline (Days 2-5, simplified into 4 plain functions)
# ===========================================================================
def condense_question(question, chat_history, llm):
    """Turn a possibly-ambiguous follow-up ('what about that?') into a standalone
    question, using the chat history. Skipped entirely if there's no history yet."""
    if not chat_history:
        return question
    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in chat_history
    )
    chain = CONDENSE_PROMPT | llm | StrOutputParser()
    return chain.invoke({"chat_history": history_text, "question": question})


def retrieve_context(store, question, k=3):
    """Get the top-k matching chunks and format them into one labeled context string."""
    docs = store.similarity_search(question, k=k)
    if not docs:
        return "No relevant documents found.", {}

    context_parts = []
    sources = {}
    for i, doc in enumerate(docs, 1):
        tag = f"Source {i}"
        sources[tag] = f"{os.path.basename(doc.metadata.get('source', '?'))} (page {doc.metadata.get('page', '?')})"
        context_parts.append(f"[{tag}]\n{doc.page_content}")
    return "\n\n".join(context_parts), sources


def generate_answer(llm, context, question):
    """The actual RAG generation step — fill the prompt, call the model, get text back."""
    chain = ANSWER_PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


def maybe_send_email(question, context):
    """If the user's message looks like an email request, let the model decide
    whether to call the email tool. Returns the tool's confirmation, or None."""
    if "email" not in question.lower() and "send" not in question.lower():
        return None

    llm_with_tools = get_llm().bind_tools([email_this])
    messages = [HumanMessage(content=f"Context:\n{context}\n\nUser request: {question}")]
    response = llm_with_tools.invoke(messages)

    if not response.tool_calls:
        return None  # model decided not to send an email after all

    tool_call = response.tool_calls[0]
    return email_this.invoke(tool_call)


# ===========================================================================
# STEP 6 — Streamlit UI
# ===========================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "store" not in st.session_state:
    st.session_state.store = load_knowledge_base_if_exists()

with st.sidebar:
    st.header("📄 Knowledge Base")

    files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

    if st.button("Build Knowledge Base", disabled=not files):
        with st.spinner("Reading, chunking, and embedding your documents..."):
            store = build_knowledge_base(files)
        if store is None:
            st.error("Couldn't extract any text from those files.")
        else:
            st.session_state.store = store
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.success("Knowledge base built!")

    st.divider()
    if st.button("New Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

st.title("📄 RAG Document Assistant")
st.caption("Simple version — ask questions about your uploaded PDFs.")

if st.session_state.store is None:
    st.info("Upload PDFs in the sidebar and click **Build Knowledge Base** to start.")
else:
    # Redraw the whole conversation so far
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for tag, label in msg["sources"].items():
                        st.write(f"**{tag}** — {label}")

    question = st.chat_input("Ask a question...")

    if question:
        # Show the user's message
        st.session_state.messages.append({"role": "user", "content": question, "sources": None})
        with st.chat_message("user"):
            st.markdown(question)

        # Run the RAG pipeline, step by step
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                llm = get_llm()

                standalone_question = condense_question(question, st.session_state.chat_history, llm)
                context, sources = retrieve_context(st.session_state.store, standalone_question)
                answer = generate_answer(llm, context, standalone_question)

                email_result = maybe_send_email(question, context)
                if email_result:
                    answer += f"\n\n📧 {email_result}"

                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        for tag, label in sources.items():
                            st.write(f"**{tag}** — {label}")

        # Save this turn to both history lists
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.chat_history.append(AIMessage(content=answer))