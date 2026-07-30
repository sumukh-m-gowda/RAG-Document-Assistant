# `app.py` — Full Code Walkthrough (Snippet by Snippet)

This document explains **every single piece** of the RAG Document Assistant's `app.py`, in the order it appears in the file. Each section shows the exact code, then explains what it does, why it's written that way, and any underlying concept a teammate would need to understand it — assuming no prior context beyond general Python.

**What this file is:** a Streamlit web app that lets a user upload PDFs, indexes them into a searchable vector database, and answers questions about them using Google's Gemini model — grounded strictly in the uploaded documents, with citations, conversational memory, and an optional "hybrid agent" mode that can send emails on request.

---

## Table of Contents
1. [Module docstring](#1-module-docstring)
2. [Imports](#2-imports)
3. [Setup & constants](#3-setup--constants)
4. [Cached resource functions](#4-cached-resource-functions)
5. [Tracking the "active" vector store collection](#5-tracking-the-active-vector-store-collection)
6. [Loading an existing vector store on startup](#6-loading-an-existing-vector-store-on-startup)
7. [Handling rate-limit errors gracefully](#7-handling-rate-limit-errors-gracefully)
8. [Batched, retry-safe embedding](#8-batched-retry-safe-embedding)
9. [Building the vector store from uploaded files](#9-building-the-vector-store-from-uploaded-files)
10. [The email-sending function (SMTP)](#10-the-email-sending-function-smtp)
11. [Turning `send_email` into an LLM-callable tool](#11-turning-send_email-into-an-llm-callable-tool)
12. [A second, tool-bound LLM instance](#12-a-second-tool-bound-llm-instance)
13. [The question-condensing prompt](#13-the-question-condensing-prompt)
14. [The grounded-answer prompt](#14-the-grounded-answer-prompt)
15. [Formatting chat history into plain text](#15-formatting-chat-history-into-plain-text)
16. [The condensing function](#16-the-condensing-function)
17. [Filtered retrieval](#17-filtered-retrieval)
18. [Formatting retrieved chunks with citations](#18-formatting-retrieved-chunks-with-citations)
19. [The hybrid-agent prompt](#19-the-hybrid-agent-prompt)
20. [`answer_question` — the central function](#20-answer_question--the-central-function)
21. [Streamlit session state initialization](#21-streamlit-session-state-initialization)
22. [Sidebar: header and API key check](#22-sidebar-header-and-api-key-check)
23. [Sidebar: file uploader and notify-email field](#23-sidebar-file-uploader-and-notify-email-field)
24. [Sidebar: the build button and its full handler](#24-sidebar-the-build-button-and-its-full-handler)
25. [Sidebar: threshold slider](#25-sidebar-threshold-slider)
26. [Sidebar: New Chat button](#26-sidebar-new-chat-button)
27. [Sidebar: hybrid agent toggle](#27-sidebar-hybrid-agent-toggle)
28. [Sidebar: chunk count caption](#28-sidebar-chunk-count-caption)
29. [Main page: title and empty state](#29-main-page-title-and-empty-state)
30. [Main page: rendering past messages](#30-main-page-rendering-past-messages)
31. [Main page: handling a new chat turn](#31-main-page-handling-a-new-chat-turn)
32. [Putting it all together — the full request lifecycle](#32-putting-it-all-together--the-full-request-lifecycle)

---

## 1. Module docstring

```python
"""
RAG Document Assistant — Streamlit App (Day 6 + Day 7)

Consolidates the logic built across all 7 days of the RAG project:
  Day 2 - Vector store (Chroma) + retrieval
  Day 3 - Full RAG chain (retrieve -> prompt -> generate)
  Day 4 - Score-threshold filtering + source citations
  Day 5 - Conversational memory (question condensing)
  Day 6 - Streamlit UI + SMTP upload notifications
  Day 7 - Hybrid agent: optional email tool-calling (Day 1 pattern) inside RAG chat

Run with:  streamlit run app.py
"""
```

**What it is:** A **module-level docstring** — a triple-quoted string placed as the very first statement in a `.py` file. Python treats it specially: it becomes the module's `__doc__` attribute, viewable via `help(app)` or `app.__doc__` if this file were imported as a module.

**Why it's here:** Pure documentation. It tells anyone opening the file (a teammate, your future self, a recruiter reviewing your GitHub) what this file does and how to run it, without needing to read any code. The "Run with: `streamlit run app.py`" line is a practical reminder — this file is **not** meant to be run with `python app.py`; Streamlit has its own launcher that sets up a local web server and manages the script's rerun behavior (explained more in section 21).

---

## 2. Imports

```python
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
```

Going through these one at a time:

| Import | What it's for |
|---|---|
| `os` | Standard library. Used for reading environment variables (`os.getenv`), building file paths (`os.path.join`), checking if files/folders exist, listing directories. |
| `re` | Standard library, "regular expressions." Used once, to pull a wait-time number out of an error message string (section 7). |
| `time` | Standard library. Used for `time.sleep()` (pausing execution) and `time.time()` (getting a unique timestamp number for naming things). |
| `smtplib` | Standard library. The actual protocol implementation for sending email via SMTP (Simple Mail Transfer Protocol) — this is what physically talks to Gmail's mail servers. |
| `email.mime.text.MIMEText` / `email.mime.multipart.MIMEMultipart` | Standard library. MIME ("Multipurpose Internet Mail Extensions") is the format email bodies are structured in. These classes build a properly formatted email message object before it's handed to `smtplib` to send. |
| `streamlit as st` | Third-party. The web app framework. Every `st.something()` call in this file renders a piece of the actual webpage — a button, a text box, a chat bubble, etc. Aliased to `st` by convention (like `pandas as pd`, `numpy as np`). |
| `dotenv.load_dotenv` | Third-party. Reads a `.env` file sitting next to this script and loads its `KEY=value` lines as environment variables, so secrets (API keys, passwords) don't have to be hardcoded into the source code. |
| `ChatGoogleGenerativeAI`, `GoogleGenerativeAIEmbeddings` | LangChain's wrapper classes for talking to two *different* Google Gemini models — one for chat/generation, one purely for turning text into embedding vectors. |
| `Chroma` | LangChain's interface to ChromaDB, the vector database used to store and search embeddings. |
| `PyPDFLoader` | Reads a PDF file off disk and turns it into LangChain `Document` objects (one per page, roughly). |
| `RecursiveCharacterTextSplitter` | Breaks long text into smaller overlapping chunks — needed because you can't (and shouldn't) hand an entire PDF to a retrieval system as one giant blob. |
| `ChatPromptTemplate` | A reusable prompt "template" with placeholders (like `{question}`) that get filled in at run time. |
| `StrOutputParser` | Strips a model's response object down to plain text — the model actually returns a richer object with metadata; this parser extracts just the `.content` string. |
| `HumanMessage`, `AIMessage` | LangChain's standard data structures representing "a message the user sent" and "a message the AI sent," used to build up conversation history. |
| `tool` | A **decorator** (explained fully in section 11) that turns a normal Python function into something an LLM can be told about and asked to call. |

**Key point for teammates less familiar with LangChain:** notice the imports are split into "LangChain" and "everything else." LangChain isn't one big library — it's split into several packages (`langchain-core`, `langchain-google-genai`, `langchain-chroma`, `langchain-community`, `langchain-text-splitters`) that each own a specific piece of functionality. This modularity is intentional on LangChain's part — you only install what you need.

---

## 3. Setup & constants

```python
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
```

**`load_dotenv()`** — this single function call scans for a file literally named `.env` in the current directory (or its parents) and loads every `KEY=value` line inside it into the process's environment variables. After this line runs, `os.getenv("GEMINI_API_KEY")` will work *as if* you'd set that environment variable manually in your terminal — but it's actually coming from the `.env` file. This is the standard pattern for keeping secrets out of source code (and out of Git, assuming `.env` is in `.gitignore`).

**`GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")`** — reads that specific variable into a Python variable for convenient reuse throughout the file. `os.getenv` returns `None` if the key doesn't exist, rather than raising an error — that's why later code checks `if not GEMINI_API_KEY:` to detect a missing key gracefully instead of crashing.

**`DATA_DIR = "data"` / `PERSIST_DIR = "chroma_db"`** — just string constants naming two folders: one where uploaded PDFs get saved, one where the vector database persists itself to disk. Using named constants instead of typing `"data"` and `"chroma_db"` repeatedly throughout the file means if you ever want to rename these folders, you change it in exactly one place.

**The comment block above `ACTIVE_COLLECTION_FILE`** — this documents a real bug that was hit and fixed during development. Originally, rebuilding the knowledge base tried to *delete* the old `chroma_db` folder before creating a new one (using `shutil.rmtree`). On Windows, this failed with a `PermissionError` because ChromaDB keeps a file handle open on its persisted files for as long as a `Chroma` client object exists in memory — and a previous one was still alive from an earlier Streamlit run. The fix (explained fully in sections 5, 6, and 9) was to stop deleting anything at all, and instead give every rebuild a uniquely named "collection" (think of it like a table inside a database) inside the *same* persistent folder.

**`ACTIVE_COLLECTION_FILE`** — a path to a small text file (`chroma_db/active_collection.txt`) whose entire job is to remember *which* collection name is the current one, so that when the app restarts, it knows which data to load.

**`DEFAULT_THRESHOLD = 0.8`** — the default similarity-score cutoff used during retrieval (fully explained in section 17). Chunks that don't score at least this close to the query get discarded rather than fed to the model.

**`st.set_page_config(...)`** — a Streamlit-specific call that must be the *first* Streamlit command in the script (a Streamlit requirement, not a Python one). It sets the browser tab's title (`page_title`), the little icon in the tab (`page_icon`, here an emoji), and `layout="wide"` which makes the page use the full browser width instead of a narrower centered column — better for a chat interface.

---

## 4. Cached resource functions

```python
@st.cache_resource(show_spinner=False)
def get_embeddings_model():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)
```

**Background concept — Streamlit's "rerun model":** this is the single most important thing to understand about how Streamlit works, and it explains *why* this caching pattern exists everywhere in the file. Streamlit does **not** work like a typical web backend where each user action triggers a small, targeted bit of server code. Instead, **every single interaction — every button click, every text input, every chat message — causes Streamlit to re-execute the entire Python script from top to bottom.** The UI you see is just the result of the most recent full run.

If that's true, then without any special handling, this line:
```python
GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
```
would run again on *every single click anywhere in the app* — reconnecting to Google's API, rebuilding the client object, over and over, for no reason. That's wasteful and slow.

**`@st.cache_resource`** is a **decorator** (a function that wraps another function to modify its behavior — more on decorators in section 11) that fixes this. The first time `get_embeddings_model()` is called, Streamlit actually runs the function body and caches the returned object. On every subsequent call — even after a full script rerun — Streamlit skips re-running the function body and just hands back the same cached object instantly. `cache_resource` (as opposed to Streamlit's other caching decorator, `cache_data`) is specifically meant for objects like API clients, database connections, and ML models — things that are expensive to create and safe to reuse, rather than plain data.

**`show_spinner=False`** — by default, `cache_resource` shows a small loading spinner in the UI the first time the function runs. Since this returns almost instantly (just constructs a client object, doesn't make a network call yet), the spinner isn't useful here, so it's turned off.

**Two separate functions for two separate models:** `get_embeddings_model()` returns a client for Google's `models/embedding-001` — a model whose *only* job is converting text into a list of numbers (a vector). `get_llm()` returns a client for `gemini-2.5-flash` — the actual chat/generation model that reads text and writes text back. These are fundamentally different kinds of models used for different stages of the pipeline (embedding happens at indexing and query time; the LLM only runs at the final generation step).

---

## 5. Tracking the "active" vector store collection

```python
def _read_active_collection():
    if os.path.isfile(ACTIVE_COLLECTION_FILE):
        with open(ACTIVE_COLLECTION_FILE, "r") as f:
            return f.read().strip()
    return None


def _write_active_collection(name):
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(ACTIVE_COLLECTION_FILE, "w") as f:
        f.write(name)
```

**Naming convention note:** the leading underscore in `_read_active_collection` and `_write_active_collection` is a Python convention (not an enforced rule) signaling "this is an internal helper function, not meant to be called from outside this file." It doesn't actually restrict access — it's purely a readability signal to other developers.

**`_read_active_collection()`** — checks whether the small tracking file exists (`os.path.isfile`), and if so, opens it, reads its entire contents as a string, and `.strip()`s off any surrounding whitespace/newlines (files often have a trailing newline character). If the file doesn't exist yet (e.g. this is a completely fresh install with nothing built), it returns `None`.

**`_write_active_collection(name)`** — the counterpart: makes sure the `chroma_db` folder exists (`os.makedirs(..., exist_ok=True)` creates the folder if missing, and does nothing — rather than raising an error — if it already exists), then opens the tracking file in write mode (`"w"`, which overwrites any previous content) and writes the new collection name into it.

**Why this exists at all:** ChromaDB, like many databases, supports multiple named "collections" inside one physical storage folder — conceptually similar to multiple tables in one SQL database. Every time the knowledge base is rebuilt (new files uploaded), this app creates a **brand new collection** with a unique name (you'll see this in section 9) instead of overwriting/deleting the old one. These two functions are how the app remembers, across restarts, *which* of those collections is the one currently in use.

---

## 6. Loading an existing vector store on startup

```python
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
```

Walking through the logic:
1. `collection_name = _read_active_collection()` — ask the tracking file which collection was last built. This could be `None` if nothing has ever been built.
2. `if collection_name and os.path.isdir(PERSIST_DIR):` — only proceed if we *both* have a remembered collection name *and* the `chroma_db` folder actually exists on disk. (`and` short-circuits — if `collection_name` is `None`/falsy, `os.path.isdir` never even runs.)
3. `Chroma(persist_directory=..., embedding_function=..., collection_name=...)` — this constructs a `Chroma` client object pointed at that specific collection inside the persisted folder. Note: this does **not** re-embed anything or make any API calls by itself — it just opens a handle to data that's already sitting on disk.
4. `store._collection.count() > 0` — a sanity check. `_collection` is Chroma's underlying low-level collection object, and `.count()` returns how many items (chunks) are stored in it. This guards against the edge case where the tracking file points to an empty or corrupted collection.
5. If everything checks out, `return store` — hand back a ready-to-query vector store.
6. Otherwise (missing tracking file, missing folder, or empty collection), `return None` — signals "there's nothing to load yet."

**Where this gets used:** called once, on every app startup (see section 21), to restore your previous session's uploaded documents without requiring you to re-upload and re-embed them every time you restart the app.

---

## 7. Handling rate-limit errors gracefully

```python
def _extract_retry_delay(error_message: str):
    """Google's 429 errors often include 'Please retry in 43.4s' — reuse that hint if present."""
    match = re.search(r"retry in ([\d.]+)s", error_message, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1  # small buffer on top of the suggested delay
    return None
```

**Background:** Google's Gemini API enforces rate limits — on the free tier, a maximum of 100 embedding requests per minute. If you exceed that, the API responds with an HTTP `429` status code ("Too Many Requests") and, helpfully, its error message often contains a specific suggestion like *"Please retry in 43.408173459s."* This function exists to **parse that number back out** of the raw error text, so the app can wait exactly as long as Google suggests instead of guessing.

**`re.search(r"retry in ([\d.]+)s", error_message, re.IGNORECASE)`** — this is a regular expression search. Breaking down the pattern:
- `retry in ` — matches that literal text.
- `([\d.]+)` — a **capture group** (the parentheses) matching one or more characters that are either digits (`\d`) or a literal period (`.`) — this captures numbers like `43.408173459`.
- `s` — matches the literal letter "s" right after the number (from "...459s").
- `re.IGNORECASE` — makes the match case-insensitive, in case the wording varies slightly.

**`if match:`** — `re.search` returns a match object if the pattern was found anywhere in the string, or `None` if it wasn't. This checks which case we're in.

**`float(match.group(1)) + 1`** — `match.group(1)` pulls out the text captured by the first (and only) parentheses group — the number itself, as a string like `"43.408173459"`. `float(...)` converts it to an actual number. The `+ 1` adds a one-second safety buffer on top of Google's suggested wait, just to be safe against timing edge cases.

**`return None`** — if the error message didn't contain that specific phrase (e.g. it's a totally different kind of error), there's nothing to extract, so the function signals "no hint available" by returning `None`. Calling code then falls back to a sensible default wait time instead.

---

## 8. Batched, retry-safe embedding

```python
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
```

This function embeds **one batch** of document chunks into the vector store, with automatic retry logic if it hits a rate limit. Let's go through it carefully.

**Parameters:**
- `store` — the `Chroma` vector store object to add documents to.
- `batch` — a list of `Document` chunks to embed and store in this call.
- `progress_callback` — an optional function. If provided, it gets *called* with status messages, so whatever code invoked this function can display live progress somewhere (in this app, that's a Streamlit UI element — see section 24). This is a common pattern: rather than this low-level function knowing anything about Streamlit, it just accepts "some function that takes a string" and calls it — keeping this function reusable and decoupled from the UI layer.
- `max_retries=2` — a default parameter value; if the caller doesn't specify how many retries to attempt, it defaults to 2.

**`for attempt in range(max_retries):`** — a loop that runs at most 2 times (`attempt` takes values `0`, then `1`).

**`try: store.add_documents(batch); return`** — attempt the actual work: ask the vector store to embed and store this batch of chunks. `store.add_documents(...)` internally calls the embedding model (bound to the store when it was created) to convert each chunk's text into a vector, then writes everything to disk. If this succeeds without throwing an exception, `return` immediately exits the function — success, no retry needed.

**`except Exception as e:`** — if *any* exception occurs during that `try` block, execution jumps here. `e` is the exception object; `Exception` is a very broad catch-all (catches almost any error type) — intentional here, since we want to inspect the error's *message text* to decide what kind of failure it was, rather than relying on catching a specific exception class (which can vary between library versions).

**`msg = str(e)`** — converts the exception object into its string representation (its error message).

**`is_rate_limit = "RESOURCE_EXHAUSTED" in msg or "429" in msg`** — a simple substring check. Google's rate-limit errors reliably contain either the text `"RESOURCE_EXHAUSTED"` or the HTTP status code `"429"` somewhere in the message. This line checks for either.

**`if not is_rate_limit or attempt == max_retries - 1: raise`** — this is the "give up" condition, checked two different ways combined with `or`:
- `not is_rate_limit` — if the error *wasn't* a rate-limit issue (some other kind of failure — bad credentials, network issue, etc.), there's no point retrying the same way, so re-raise immediately.
- `attempt == max_retries - 1` — if this was the *last* allowed attempt (remember, `attempt` goes 0, 1 for `max_retries=2`, so `max_retries - 1` is `1`, the final iteration), also give up rather than looping forever.
- `raise` (with no argument) — re-raises the *currently being handled* exception, preserving its original type and traceback. This propagates the error up to whatever called this function, to be handled there (see section 24, where the button handler catches it and shows a friendly message).

**`wait = min(_extract_retry_delay(msg) or 30, 65)`** — figures out how long to pause before retrying:
- `_extract_retry_delay(msg) or 30` — try to get Google's suggested wait time (section 7); if that returns `None` (no hint found), fall back to a default of 30 seconds. (This relies on `None` being "falsy" in Python — `X or Y` evaluates to `Y` when `X` is `None`, `False`, `0`, or empty.)
- `min(..., 65)` — cap whatever we got at 65 seconds maximum, so a single retry never blocks for an unreasonably long time even if the suggested delay were huge.

**`if progress_callback: progress_callback(f"...")`** — if the caller gave us a progress-reporting function, call it with a human-readable status message, including an f-string (`f"..."`) that embeds the wait time and which attempt number this is.

**`time.sleep(wait)`** — actually pause execution for `wait` seconds before the loop goes back around and tries again.

**`raise RuntimeError("Failed to embed after retries due to API rate limits.")`** — this line is only reached if the `for` loop completes all its iterations *without* ever successfully returning or re-raising inside the loop — which, given the logic above, is actually a defensive fallback that shouldn't normally trigger (since the `raise` inside the except block on the last attempt would fire first), but it's good practice to have an explicit final error rather than letting the function silently fall through and return `None`.

**Big-picture takeaway:** this is a standard **retry with backoff** pattern, extremely common whenever code talks to external, rate-limited services. The key ideas — catch the error, check if it's the *specific kind* of error worth retrying, wait an appropriate amount, retry a limited number of times, then give up cleanly — are reusable well beyond this specific project.

---

## 9. Building the vector store from uploaded files

```python
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
```

**`Chroma(persist_directory=..., embedding_function=..., collection_name=...)`** — creates a brand-new (initially empty) Chroma collection at the given name, inside the persisted folder. No documents are added yet at this point — that happens in the loop below.

**`total = len(chunks)`** — how many chunks total need to be embedded.

**`for i in range(0, total, batch_size):`** — this is Python's pattern for **iterating in fixed-size steps**. `range(start, stop, step)` with `step=batch_size` (50 by default) produces `0, 50, 100, 150, ...` up to (but not including) `total`. So `i` represents the starting index of each batch.

**`batch = chunks[i:i + batch_size]`** — Python list **slicing**: grabs the sub-list from index `i` up to (but not including) `i + batch_size`. On the final iteration, if there are fewer than `batch_size` chunks remaining, slicing simply returns however many are left — no out-of-bounds error, since Python slicing gracefully clamps to the list's actual length.

**`progress_callback(f"Embedding chunks {i + 1}-{min(i + batch_size, total)} of {total}...")`** — reports progress. `i + 1` converts from 0-indexed to a human-friendly 1-indexed number. `min(i + batch_size, total)` makes sure the displayed "end" number doesn't overshoot the actual total on the last (possibly partial) batch.

**`_add_batch_with_retry(store, batch, progress_callback=progress_callback)`** — calls the retry-safe function from section 8 to actually embed and store this one batch.

**`if i + batch_size < total: time.sleep(pause_seconds)`** — after each batch *except the last one*, pause briefly (2 seconds by default) before moving to the next batch. This spreads out the requests over time, further reducing the chance of tripping the per-minute rate limit — combined with the retry logic in section 8, this gives two layers of protection: proactive pacing, and reactive backoff if a limit is still hit.

**`return store`** — hands back the now-populated vector store.

---

```python
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
```

This is the top-level function that runs the **entire ingestion pipeline** — from raw uploaded files to a queryable vector store. It's called from the "Build / Rebuild Knowledge Base" button (section 24).

**`os.makedirs(DATA_DIR, exist_ok=True)`** — ensures the `data/` folder exists before we try to save files into it.

**Saving uploaded files to disk:**
```python
saved_paths = []
for uf in uploaded_files:
    path = os.path.join(DATA_DIR, uf.name)
    with open(path, "wb") as f:
        f.write(uf.getbuffer())
    saved_paths.append(path)
```
`uploaded_files` is a list of Streamlit's special `UploadedFile` objects (produced by `st.file_uploader` — see section 23), representing files the user selected in their browser but which only exist in memory/temporarily so far — not yet on disk. For each one (`uf`):
- `os.path.join(DATA_DIR, uf.name)` — builds a proper file path combining the `data/` folder with the file's original name, using `os.path.join` rather than manual string concatenation because it correctly handles path separators across operating systems (`/` on Mac/Linux, `\` on Windows).
- `open(path, "wb")` — opens (creates) a file at that path in **write-binary** mode (`"wb"`) — binary because PDF files are binary data, not plain text, so we must not attempt any text encoding/decoding.
- `f.write(uf.getbuffer())` — `uf.getbuffer()` gets the raw bytes of the uploaded file from Streamlit's internal buffer, and `.write()` saves those bytes to the file on disk.
- `with open(...) as f:` — the `with` statement is a **context manager**; it guarantees the file gets properly closed afterward, even if an error occurs partway through, without needing an explicit `f.close()` call.
- `saved_paths.append(path)` — keeps track of every file path we just saved, in a plain Python list, for use in the next step.

**This is also where the earlier accumulation bug was fixed.** An earlier version of this function scanned the *entire* `data/` folder (`os.listdir(DATA_DIR)`) for PDFs to load, which meant every file ever uploaded across every session got re-embedded on every rebuild. The current version only loads from `saved_paths` — the specific files from *this* upload — so old leftover files sitting in `data/` are simply ignored.

**Loading the PDFs into `Document` objects:**
```python
documents = []
for path in saved_paths:
    documents.extend(PyPDFLoader(path).load())
```
For each saved file path, `PyPDFLoader(path)` creates a loader object for that specific PDF, and `.load()` actually reads it and returns a list of LangChain `Document` objects (typically one per page — each holding the page's raw text plus metadata like the source filename and page number). `documents.extend(...)` — as opposed to `.append(...)` — is used because `.load()` returns a *list*, and `extend` adds each item of that list individually into the `documents` list, rather than nesting a list-inside-a-list (which `.append` would do).

**`if not documents: return None, 0`** — a guard clause. If no text could be extracted at all (e.g. the "PDF" was actually empty, corrupted, or an image-only scan with no extractable text), there's nothing to build an index from. The function returns a tuple `(None, 0)` — `None` for "no store was built" and `0` for "zero chunks" — signaling failure to the caller in a way that's easy to check (`if store is None: ...`).

**Splitting into chunks:**
```python
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
```
`chunk_size=1000` means each chunk targets roughly 1000 characters. `chunk_overlap=200` means consecutive chunks share 200 characters at their boundary, so an idea or sentence that would otherwise get awkwardly split in half is more likely to appear complete in at least one chunk. "Recursive" refers to the splitting strategy: it tries to break on paragraph boundaries first, then sentences, then words, only falling back to a hard character cut as a last resort — this keeps chunks more semantically coherent than naive fixed-length slicing.

**Generating a unique collection name:**
```python
collection_name = f"docs_{int(time.time())}"
```
`time.time()` returns the current time as a floating-point number of seconds since the "Unix epoch" (January 1, 1970) — e.g. `1732481920.583...`. `int(...)` truncates it to a whole number of seconds. The f-string wraps it into a name like `"docs_1732481920"`. Because this number changes every second, it's virtually guaranteed to be unique for each rebuild, giving every rebuild its own isolated Chroma collection (as explained in section 3's comment block) — no deletion of prior data required.

**`store = build_vector_store_batched(chunks, collection_name, progress_callback=progress_callback)`** — runs the actual batched embedding process from section 9's first part.

**`_write_active_collection(collection_name)`** — remembers this new collection as the "current" one, so future app restarts load it automatically (section 6).

**`return store, len(chunks)`** — hands back both the ready-to-query store object and the total number of chunks that were indexed, for display in the UI.

---

## 10. The email-sending function (SMTP)

```python
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
```

**Type hints in the signature:** `recipient_email: str, subject: str, body: str) -> str` — these `: str` annotations are Python **type hints**. They don't enforce anything at runtime by themselves (Python won't stop you from passing an integer), but they document the expected types for humans and tools (IDEs, linters), and — importantly for this project — they're also read by LangChain's `@tool` decorator (section 11) to build a schema telling the LLM exactly what arguments this function expects.

**`sender_email = os.environ.get('SENDER_EMAIL')` / `app_password = os.environ.get('SENDER_APP_PASSWORD')`** — reads two more secrets from environment variables (loaded from `.env` via `load_dotenv()` earlier). `SENDER_APP_PASSWORD` specifically needs to be a Gmail **App Password** — a 16-character token you generate specifically for scripts/apps, rather than your real Gmail password, because Google blocks direct password logins from non-browser clients for security reasons. An App Password can be individually revoked without changing your main password if it's ever compromised.

**Building the email message:**
```python
msg = MIMEMultipart()
msg['From'] = sender_email
msg['To'] = recipient_email
msg['Subject'] = subject
msg.attach(MIMEText(body, 'plain'))
```
`MIMEMultipart()` creates a container object representing the overall email message (capable of holding multiple parts, like a plain-text body plus attachments, though only one plain-text part is used here). Setting `msg['From']`, `msg['To']`, `msg['Subject']` populates the standard email headers by assigning into what behaves like a dictionary. `MIMEText(body, 'plain')` wraps the actual message text, tagging it as plain text (as opposed to HTML), and `.attach(...)` adds it as a part of the overall message.

**Sending it:**
```python
with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
    server.login(sender_email, app_password)
    server.sendmail(sender_email, recipient_email, msg.as_string())
```
`smtplib.SMTP_SSL('smtp.gmail.com', 465)` opens an **SSL-encrypted connection** to Gmail's SMTP server on port 465 (the standard port for encrypted SMTP submission). Again, `with ... as server:` is a context manager — it guarantees the network connection gets properly closed afterward regardless of success or failure. `server.login(...)` authenticates using the sender's address and App Password. `server.sendmail(sender_email, recipient_email, msg.as_string())` actually transmits the message — `msg.as_string()` converts the structured `MIMEMultipart` object into the raw text format that the SMTP protocol expects on the wire.

**Error handling — three levels of specificity:**
```python
except smtplib.SMTPAuthenticationError:
    return '...'
except smtplib.SMTPRecipientsRefused:
    return f'...'
except Exception as e:
    return f'Failed to send email: {str(e)}'
```
Python checks `except` clauses **in order**, and only the first matching one runs:
- `SMTPAuthenticationError` — specifically means the login credentials were rejected (wrong App Password, or accidentally using a real password). The error message tells the user exactly what's likely wrong.
- `SMTPRecipientsRefused` — specifically means the server rejected the *destination* address (e.g. a malformed or nonexistent email address).
- `Exception as e` — a catch-all for absolutely anything else that could go wrong (network failure, unexpected server response, etc.), with the raw error message included for debugging.

**Why every branch `return`s a string instead of raising/crashing:** this matters a lot in the context of this app (and was equally true in the original standalone email agent project). If this function threw an exception on failure, and it's later called as a *tool* by an LLM agent (section 11), an uncaught exception would break the whole request. By always returning a descriptive string — success or failure — the calling code (and eventually the LLM itself, in agent mode) can read and react to what happened, rather than the whole app crashing.

---

## 11. Turning `send_email` into an LLM-callable tool

```python
@tool
def email_this(recipient_email: str, subject: str, message: str) -> str:
    """
    Send an email to the given recipient with the given subject and message.
    Use this ONLY when the user explicitly asks to email, send, or mail something
    to a specific address (e.g. "email this summary to alex@gmail.com"). Never use
    this tool unless the user clearly asked for an email to be sent.
    """
    return send_email(recipient_email, subject, message)
```

**Background — what is a decorator?** In Python, `@something` written directly above a function definition is special syntax meaning "pass this function through `something` and replace it with whatever `something` returns." It's exactly equivalent to writing:
```python
def email_this(recipient_email, subject, message):
    ...
email_this = tool(email_this)
```
`@tool` is a decorator provided by LangChain (`from langchain_core.tools import tool`, imported in section 2). It takes an ordinary Python function and wraps it into a special `Tool` object that LangChain (and, downstream, the LLM) knows how to work with.

**What information does `@tool` extract from this function, and how?**
- **The name** — taken directly from the function's name: `"email_this"`. This is the identifier the model will use when it wants to request a call to this specific tool.
- **The description** — taken directly from the **docstring**. This is arguably the most important part: the model never reads this function's actual code. It only ever sees the *name* and this *description text* when deciding whether and how to use the tool. That's why the docstring here is written as an instruction aimed at the model ("Use this ONLY when the user explicitly asks...") rather than just describing what the code does — it's functioning as guidance for an AI decision-maker, not just human documentation.
- **The arguments schema** — built from the function's parameter names and type hints: `recipient_email: str`, `subject: str`, `message: str`. This tells the model exactly what pieces of information it needs to supply, and in what format, when it requests a call to this tool.

**The function body — `return send_email(recipient_email, subject, message)`** — notice this is a thin wrapper. All the actual SMTP logic lives in the plain `send_email` function from section 10; `email_this` just forwards its arguments straight through. This separation exists so that `send_email` remains a normal, independently testable/callable Python function (e.g. it's also called directly, without going through the LLM, in section 24's upload notification feature), while `email_this` is specifically the *tool-shaped* wrapper used only in the LLM agent path.

---

## 12. A second, tool-bound LLM instance

```python
@st.cache_resource(show_spinner=False)
def get_llm_with_tools():
    """Same Gemini model as get_llm(), but with the email tool bound — this is the
    Day 1 tool-calling pattern (bind_tools) applied inside the RAG app."""
    return get_llm().bind_tools([email_this])
```

**`get_llm().bind_tools([email_this])`** — `get_llm()` (section 4) returns the cached base chat model. `.bind_tools([email_this])` returns a **new** model object — conceptually a "copy" of the original model, but configured so that every request sent through it automatically includes the schema (name, description, arguments) of every tool in the list — here, just `email_this`. This is what makes the model "tool-aware": without `bind_tools`, the model has no way of knowing this function exists at all.

**Why keep this as a *separate* cached function from `get_llm()`, rather than always binding tools to the one model?** Because tool-awareness is optional in this app (see the sidebar checkbox in section 27) — plain question-answering should behave exactly as it did before tools were introduced, with zero chance of the model unexpectedly trying to call a tool. Keeping two distinct model handles — one plain, one tool-bound — makes that separation explicit and impossible to accidentally mix up.

**Why `@st.cache_resource` again here:** same reasoning as section 4 — constructing this bound-model object is cheap but unnecessary to redo on every script rerun, so it's cached exactly once.

---

## 13. The question-condensing prompt

```python
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
```

**`ChatPromptTemplate.from_template(...)`** — creates a reusable prompt template out of a plain multi-line string. The curly-brace placeholders — `{chat_history}` and `{question}` — are **not** Python f-string interpolation (this isn't an f-string; there's no `f` prefix). Instead, they're LangChain's own template syntax, filled in later when the template is actually *invoked* with a dictionary of values (you'll see this in section 16: `chain.invoke({"chat_history": ..., "question": ...})`).

**Why this prompt exists — the core problem it solves:** vector-based retrieval only ever looks at the *literal text* of whatever query it's given — it has no awareness of prior conversation turns. If a user first asks "What is LangChain?" and then follows up with "How does it handle memory?", searching the vector store for the literal text "How does it handle memory?" would fail, because the word "it" carries the entire meaning and the retriever has no way to resolve what "it" refers to. This prompt's job is to have the LLM **rewrite** an ambiguous follow-up into a fully self-contained question (e.g. "How does LangChain handle memory?") *before* any retrieval happens — using the conversation history as context to resolve the ambiguity.

**"Do not answer the question - only rewrite it."** — this line matters. Without it, the model might get "helpful" and just answer directly instead of returning a rewritten question, which would break the pipeline (the calling code expects a *question* back, not an *answer*, at this stage).

---

## 14. The grounded-answer prompt

```python
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
```

This is the prompt used for **normal (non-agent-mode) question answering** — the core RAG generation step. Each rule earns its place:

- **"using ONLY the context provided below" / "Base your answer strictly on the context. Do not use outside knowledge."** — this is the **grounding instruction**, arguably the single most important line in the entire prompt. Without it, even when given relevant retrieved context, LLMs will often blend in information from their own general training knowledge, which defeats the purpose of RAG (you specifically want answers traceable back to *your* documents, not the model's possibly-outdated or hallucinated general knowledge).
- **"Reference the relevant [Source N] tag(s) inline..."** — instructs the model to actively cite which piece of context it drew from, using the `[Source N]` labels that get attached to each chunk (explained in section 18). This is what allows a user to verify an answer against the original document.
- **The `"NO_RELEVANT_CONTEXT"` rule** — a specific, exact-string instruction for a specific, exact scenario: when retrieval (section 17) found nothing similar enough to the question, the formatting function (section 18) sets `context` to the literal string `"NO_RELEVANT_CONTEXT"`. This rule tells the model precisely how to respond in that case — a clean, honest, single-sentence refusal — rather than trying to be "helpful" and guessing an answer from irrelevant or absent context.
- **"You may refer naturally to earlier parts of the conversation, but never invent facts..."** — allows conversational tone (e.g. "Building on that...") while keeping the grounding rule airtight; conversational *style* and factual *grounding* are two independent concerns, and this line makes clear that relaxing the former doesn't relax the latter.
- **"Be concise and direct."** — a general quality/style instruction, keeping answers focused rather than verbose.

---

## 15. Formatting chat history into plain text

```python
def format_history(chat_history):
    if not chat_history:
        return "(no previous conversation)"
    lines = []
    for msg in chat_history:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)
```

**`if not chat_history: return "(no previous conversation)"`** — an empty list is "falsy" in Python, so `not chat_history` is `True` when the list is empty (i.e. this is the very first message in a session). In that case, there's nothing to summarize, so a simple placeholder string is returned instead of an empty block of text (which could confuse the prompt template).

**The loop:**
```python
lines = []
for msg in chat_history:
    role = "User" if isinstance(msg, HumanMessage) else "Assistant"
    lines.append(f"{role}: {msg.content}")
```
`chat_history` is a list alternating between `HumanMessage` and `AIMessage` objects (LangChain's structured message types, imported in section 2). For each message:
- `isinstance(msg, HumanMessage)` — checks the message's actual Python type. If it's a `HumanMessage`, label it `"User"`; otherwise (it must be an `AIMessage`), label it `"Assistant"`. This is a **ternary conditional expression** (`X if condition else Y`) — a compact one-line if/else.
- `f"{role}: {msg.content}"` — builds a line like `"User: What is LangChain?"`, accessing `.content`, the attribute where these message objects store their actual text.
- `lines.append(...)` — collects each formatted line into a plain Python list.

**`return "\n".join(lines)`** — combines the list of individual lines into one single string, with a newline character (`\n`) inserted between each one — producing a readable, multi-line transcript.

**Why this function exists:** the `CONDENSE_PROMPT` template (section 13) needs `{chat_history}` filled in as a single plain-text string, but the actual conversation is stored internally as a structured list of typed message objects (useful for programmatic access elsewhere in the app). This function is the bridge between those two representations.

---

## 16. The condensing function

```python
def condense_question(question, chat_history, llm):
    if not chat_history:
        return question
    chain = CONDENSE_PROMPT | llm | StrOutputParser()
    return chain.invoke({"chat_history": format_history(chat_history), "question": question}).strip()
```

**`if not chat_history: return question`** — an important optimization/correctness check: if there's no prior conversation, there's nothing to "condense" against — the question is already standalone by definition. Returning early here also **saves an unnecessary LLM API call** on every very first message of a session.

**`chain = CONDENSE_PROMPT | llm | StrOutputParser()`** — this is **LangChain Expression Language (LCEL)**. The `|` (pipe) operator chains steps together, where each step's output becomes the next step's input — conceptually similar to Unix shell pipes (`cat file | grep pattern | sort`). Reading left to right:
1. `CONDENSE_PROMPT` — takes a dictionary of template variables and produces a fully filled-in prompt.
2. `llm` — takes that prompt and sends it to the Gemini model, producing a structured response object.
3. `StrOutputParser()` — takes that structured response and extracts just the plain-text `.content` string.

**`chain.invoke({"chat_history": format_history(chat_history), "question": question})`** — actually runs the chain, providing the values for the template's placeholders. `format_history(chat_history)` (section 15) converts the structured history into a plain-text transcript first.

**`.strip()`** — removes any leading/trailing whitespace from the model's response, since LLM outputs sometimes include stray newlines or spaces.

**Design note — this is a deliberately separate, smaller LLM call.** The condensing step and the final answer-generation step are two independent calls to the model, each with one narrow, focused job. This is a common and useful pattern in LLM application design: rather than asking one prompt to do multiple things at once ("rewrite this AND answer it AND decide if a tool is needed"), breaking complex behavior into a sequence of smaller, single-purpose steps tends to produce more reliable results.

---

## 17. Filtered retrieval

```python
def retrieve_filtered(store, query, k=5, threshold=DEFAULT_THRESHOLD):
    results = store.similarity_search_with_score(query, k=k)
    return [(doc, score) for doc, score in results if score <= threshold]
```

**`store.similarity_search_with_score(query, k=k)`** — this is Chroma's core retrieval operation. Internally, it: (1) embeds the `query` text into a vector, using the same embedding model that was bound to this store when it was created; (2) compares that query vector against every stored chunk vector; (3) returns the `k` closest matches (5, by default), each paired with a **distance score**. For Chroma's default distance metric, **lower score = more similar** (it's measuring distance/dissimilarity, not a 0-100% similarity percentage).

**`return [(doc, score) for doc, score in results if score <= threshold]`** — this is a **list comprehension**, a compact way to build a new list by filtering/transforming an existing one. Unpacked into an equivalent explicit loop, it reads:
```python
filtered = []
for doc, score in results:
    if score <= threshold:
        filtered.append((doc, score))
return filtered
```
`results` is a list of `(document, score)` tuples. `for doc, score in results` unpacks each tuple into its two components on each iteration. `if score <= threshold` keeps only the pairs whose distance score is *at or below* the threshold — i.e., genuinely close matches — discarding weaker matches even if they technically made it into the top-`k`.

**Why filtering matters:** without it, even a completely unrelated question would still get *some* chunks back (since `similarity_search` always returns its top-`k`, regardless of whether any of them are actually relevant) — and an LLM handed *any* text, even barely-relevant text, will often try to construct an answer from it rather than admit the text doesn't help. Filtering removes weak matches *before* they ever reach the prompt, which is what makes the `"NO_RELEVANT_CONTEXT"` fallback (sections 14, 18) actually trigger correctly when appropriate.

---

## 18. Formatting retrieved chunks with citations

```python
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
```

**`if not filtered_results: return "NO_RELEVANT_CONTEXT", {}`** — if, after filtering, nothing survived (either the vector search found nothing close at all, or everything found was filtered out for being too dissimilar), this function returns the exact sentinel string `"NO_RELEVANT_CONTEXT"` — the specific value the `ANSWER_PROMPT` (section 14) is instructed to watch for — paired with an empty citation dictionary (`{}`), since there are no sources to cite.

**The main loop:**
```python
context_parts, citation_map = [], {}
for i, (doc, score) in enumerate(filtered_results, 1):
    tag = f"Source {i}"
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", "?")
    citation_map[tag] = f"{os.path.basename(source)} (page {page})"
    context_parts.append(f"[{tag}]\n{doc.page_content}")
```
- `context_parts, citation_map = [], {}` — initializes two empty containers in one line: a list to build the eventual context text, and a dictionary to build a lookup from a citation tag to a human-readable source description.
- `enumerate(filtered_results, 1)` — `enumerate` pairs each item in an iterable with an index; the second argument, `1`, means "start counting from 1" instead of Python's default of 0 — so citations read naturally as "Source 1", "Source 2", ... rather than starting at "Source 0".
- `tag = f"Source {i}"` — builds the citation label for this chunk.
- `doc.metadata.get("source", "unknown")` / `doc.metadata.get("page", "?")` — pulls the original filename and page number out of the chunk's metadata (attached automatically back when `PyPDFLoader` first loaded the document — section 9). `.get(key, default)` safely returns a fallback value (`"unknown"` or `"?"`) instead of raising an error if that metadata key happens to be missing.
- `citation_map[tag] = f"{os.path.basename(source)} (page {page})"` — stores a human-readable description for this tag. `os.path.basename(source)` strips any folder path down to just the filename (e.g. turning `"data/report.pdf"` into `"report.pdf"`), since the user doesn't need to see the internal folder structure.
- `context_parts.append(f"[{tag}]\n{doc.page_content}")` — builds one block of context text: the citation tag on its own line (e.g. `[Source 1]`), followed by the chunk's actual text content.

**`return "\n\n".join(context_parts), citation_map`** — joins all the individual context blocks together, separated by a blank line (`"\n\n"`) between each, into one big context string — and returns it alongside the citation lookup dictionary. This function returns **two values** as a tuple, which is why every call site does `context, citation_map = format_docs_with_citations(...)` — Python automatically unpacks a returned tuple into multiple variables when the left-hand side has matching structure.

---

## 19. The hybrid-agent prompt

```python
HYBRID_INSTRUCTIONS = """You are a helpful assistant answering questions using ONLY the context provided below.

Rules:
- Base your answer strictly on the context. Do not use outside knowledge.
- Reference the relevant [Source N] tag(s) inline when you use information from them.
- If the context is exactly "NO_RELEVANT_CONTEXT", respond only with:
  "I don't have relevant information in the provided documents to answer that."
- You may refer naturally to earlier parts of the conversation, but never invent
  facts that aren't in the context.
- Be concise and direct.
- You have access to an "email_this" tool. Use it ONLY if the user explicitly asks
  you to email, send, or mail something to a specific address. Otherwise, just answer
  normally in text — do not send an email unless clearly asked to.

Context:
{context}

Question:
{question}
"""
```

**Note the type difference from `ANSWER_PROMPT`:** this is a **plain Python string** (specifically, filled in manually via `.format()`, as you'll see in section 20), *not* a `ChatPromptTemplate` object. This is a deliberate implementation choice: because the tool-calling path (section 20) needs to build a raw list of `HumanMessage` objects to pass directly to the tool-bound model — rather than running through a `ChatPromptTemplate | llm | parser` LCEL chain — a plain string that supports `.format(context=..., question=...)` is simpler to work with in that specific code path.

**Content-wise**, this prompt is identical to `ANSWER_PROMPT` (section 14) for the first five rules — same grounding instruction, same citation instruction, same `"NO_RELEVANT_CONTEXT"` handling, same conversational-but-grounded rule, same conciseness rule — **plus one new rule at the end** specifically about the `email_this` tool:

**"You have access to an 'email_this' tool. Use it ONLY if the user explicitly asks you to email, send, or mail something to a specific address... do not send an email unless clearly asked to."** — this is a **tool-use guardrail**. It's worth understanding why this is necessary in addition to (not instead of) the tool's own docstring (section 11), which already says something similar. Belt-and-suspenders instruction repetition like this is a common, practical pattern in prompt engineering — reinforcing an important behavioral constraint in more than one place tends to make the model more reliably follow it, especially for actions with real-world side effects (sending an actual email) where an unwanted/accidental tool call would be a meaningfully bad outcome, not just a wrong text answer.

---

## 20. `answer_question` — the central function

```python
def answer_question(store, llm, question, chat_history, threshold, agent_mode=False):
    standalone = condense_question(question, chat_history, llm)
    filtered = retrieve_filtered(store, standalone, threshold=threshold)
    context, citation_map = format_docs_with_citations(filtered)

    if not agent_mode:
        chain = ANSWER_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": standalone})
        return answer, citation_map, standalone

    # --- Hybrid agent path (Day 7) ---
    # Same shape as Day 1's run_agent(): send messages -> check tool_calls -> if
    # present, execute the tool and feed the result back -> ask for a final answer.
    llm_tools = get_llm_with_tools()
    prompt_text = HYBRID_INSTRUCTIONS.format(context=context, question=standalone)
    messages = [HumanMessage(content=prompt_text)]

    response = llm_tools.invoke(messages)
    messages.append(response)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_result = email_this.invoke(tool_call)
            messages.append(tool_result)
        final = llm_tools.invoke(messages)
        answer = final.content
    else:
        answer = response.content

    return answer, citation_map, standalone
```

This function ties together nearly everything explained so far — it's called once per user message (section 31), and it's the single place where the entire question-answering pipeline runs, in either of its two modes.

**The shared first three lines** (run regardless of mode):
```python
standalone = condense_question(question, chat_history, llm)
filtered = retrieve_filtered(store, standalone, threshold=threshold)
context, citation_map = format_docs_with_citations(filtered)
```
1. Resolve any ambiguous pronouns/references in the raw `question` using conversation history (section 16), producing `standalone`.
2. Run filtered similarity search against the vector store using that standalone question (section 17).
3. Format whatever survived filtering into a citation-labeled context string, plus a lookup table (section 18).

At this point, regardless of mode, we have `context` (either real retrieved text, or the sentinel `"NO_RELEVANT_CONTEXT"`) and `citation_map` ready.

**Branch 1 — `if not agent_mode:` (the default, simpler path):**
```python
chain = ANSWER_PROMPT | llm | StrOutputParser()
answer = chain.invoke({"context": context, "question": standalone})
return answer, citation_map, standalone
```
This is a straightforward LCEL chain — identical in structure to the one in `condense_question` (section 16), just using `ANSWER_PROMPT` instead. It fills in the template, sends it to the plain (non-tool-bound) model, extracts the text, and returns immediately. Three values are returned: the answer text, the citation lookup, and the standalone (condensed) question — this last one gets used later purely for display purposes (showing the user "interpreted as: ...", section 31).

**Branch 2 — the hybrid agent path (only runs when `agent_mode=True`):**

```python
llm_tools = get_llm_with_tools()
```
Grab the cached, tool-bound model (section 12) instead of the plain one.

```python
prompt_text = HYBRID_INSTRUCTIONS.format(context=context, question=standalone)
messages = [HumanMessage(content=prompt_text)]
```
Build the full instruction text using Python's `.format()` string method (filling in `{context}` and `{question}` from section 19's template), then wrap it in a `HumanMessage` object and place it as the first (and so far only) entry in a `messages` list. This `messages` list represents the running conversation being sent to the model at each step — this pattern (a growing list of message objects) should look familiar if you've worked with the original Day 1 email agent project; it's the exact same underlying structure.

```python
response = llm_tools.invoke(messages)
messages.append(response)
```
Send the message list to the tool-aware model. The `response` object comes back — it could contain either plain text, or a *request* to call a tool (or potentially both/neither, depending on the model's decision). Regardless of what it contains, it gets appended to `messages`, so the model's own prior output becomes part of the context for the next step (if there is one).

```python
if response.tool_calls:
```
`response.tool_calls` is a list attribute on the response object — LangChain populates it with details of any tool call(s) the model decided to request. If the model didn't request any tool, this list is empty, which is "falsy" in Python — so `if response.tool_calls:` is only `True` when the model actually wants to call something.

**If a tool call was requested:**
```python
for tool_call in response.tool_calls:
    tool_result = email_this.invoke(tool_call)
    messages.append(tool_result)
final = llm_tools.invoke(messages)
answer = final.content
```
- `for tool_call in response.tool_calls:` — loops through every requested tool call (in this app, realistically always just one, since only one tool exists and the prompt asks for at most one email action per turn, but the loop handles the general case).
- `email_this.invoke(tool_call)` — this is the moment the **actual email gets sent**. `tool_call` is a dictionary-like structure containing the tool's name and the specific arguments the model filled in (recipient, subject, message — extracted by the model from the user's natural-language request). `.invoke(...)` on a `@tool`-decorated function runs the real underlying Python code (section 11 → section 10 → real SMTP call) and wraps the return value in a `ToolMessage` object.
- `messages.append(tool_result)` — adds that tool's result (e.g. `"Email successfully sent to..."` or an error string) into the conversation history, so the model can see what actually happened.
- `final = llm_tools.invoke(messages)` — calls the model **again**, now with the full history including the tool result, so it can generate a natural-language confirmation response (e.g. *"I've sent the summary to alex@gmail.com."*) rather than just returning raw tool output to the user.
- `answer = final.content` — extracts the plain text of that final confirmation.

**If no tool call was requested:**
```python
else:
    answer = response.content
```
The model just answered directly in text (the normal case for any question that isn't asking for an email to be sent) — use that text as-is.

**`return answer, citation_map, standalone`** — same three-value return shape as branch 1, so the calling code (section 31) doesn't need to know or care which branch actually ran.

**This entire tool-call handling block mirrors the "agent loop" pattern from the original Day 1 project almost exactly:** send messages → check if a tool was requested → if so, execute it and feed the result back → get a final response. The only structural difference is that Day 1's loop used `while True` to allow for an arbitrary, unbounded number of sequential tool calls across multiple turns, whereas here it's a single check-and-respond pass, since only one tool exists and at most one email action makes sense per user turn.

---

## 21. Streamlit session state initialization

```python
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: {role, content, sources}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of HumanMessage/AIMessage for condensing
if "store" not in st.session_state:
    st.session_state.store = load_existing_store()
```

**Recall from section 4: Streamlit reruns the entire script top-to-bottom on every interaction.** If this were just plain Python variables (`messages = []`), they'd be wiped out and recreated as empty on *every single rerun* — meaning the chat would appear to forget everything after each message, which would make a chat interface completely unusable.

**`st.session_state`** is Streamlit's solution: a special, dictionary-like object that **persists across reruns**, scoped to the current browser session/tab. Think of it as the app's "memory" that survives the constant re-execution of the script.

**The pattern `if "key" not in st.session_state: st.session_state.key = initial_value`** is the standard Streamlit idiom for **initialize-once** state. Here's why the `if` check matters: without it, every rerun would reset `st.session_state.messages` back to `[]`, erasing chat history on every interaction — defeating the entire purpose. With the check, the initialization code only actually executes on the *very first* run of a session (when the key genuinely doesn't exist yet); on every subsequent rerun, the condition is `False`, so the existing value in `session_state` is left completely untouched.

**Three separate pieces of state, each with a distinct purpose:**
- `st.session_state.messages` — a list of plain Python dictionaries, each shaped like `{"role": "user"/"assistant", "content": "...", "sources": {...} or None}`. This is what actually gets *rendered* in the chat UI (section 30) — it includes extra display-only information (like the sources dictionary) that doesn't belong in the LLM-facing conversation history.
- `st.session_state.chat_history` — a list of `HumanMessage`/`AIMessage` objects specifically used to feed `condense_question` (section 16). This is a separate, more minimal representation containing only what the condensing LLM call actually needs.
- `st.session_state.store` — the actual `Chroma` vector store object. Initialized by calling `load_existing_store()` (section 6) — meaning if a previous knowledge base already exists on disk from an earlier session, it gets loaded automatically without requiring the user to re-upload anything.

**Why two separate history representations (`messages` vs. `chat_history`) instead of one?** Separation of concerns: the UI-rendering code (section 30) shouldn't need to know anything about `HumanMessage`/`AIMessage` internals, and the condensing logic (section 16) shouldn't need to deal with UI-specific fields like `sources`. Keeping them as two independently-maintained lists (both updated together at the end of each turn, section 31) keeps each piece of code focused on exactly what it needs.

---

## 22. Sidebar: header and API key check

```python
with st.sidebar:
    st.header("📄 Knowledge Base")

    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY not found. Add it to your .env file.")
```

**`with st.sidebar:`** — another context manager, but used here purely for its *layout* effect rather than for resource cleanup: any Streamlit UI element created inside this `with` block gets placed in the collapsible sidebar panel on the left of the page, rather than in the main page area.

**`st.header("📄 Knowledge Base")`** — renders a header-styled piece of text in the sidebar.

**`if not GEMINI_API_KEY: st.error(...)`** — recall from section 3 that `GEMINI_API_KEY` could be `None` if the `.env` file is missing or doesn't contain that key. `not None` is `True`, so this check catches that case and displays a red error box directly in the app UI, telling the user exactly what's wrong and how to fix it — much friendlier than letting the app silently fail somewhere deeper in the code with a cryptic API error.

---

## 23. Sidebar: file uploader and notify-email field

```python
    uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

    notify_email = st.text_input(
        "Notify this email on upload",
        value=os.environ.get("SENDER_EMAIL", ""),
        help="Sends a confirmation email via Gmail SMTP once the knowledge base is built."
    )
```

**`st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)`** — renders a drag-and-drop / click-to-browse file upload widget. `type=["pdf"]` restricts the browser's file picker to only show/accept `.pdf` files. `accept_multiple_files=True` allows selecting more than one file at once. The return value, `uploaded_files`, is a Python list of Streamlit's `UploadedFile` objects (or an empty list if nothing's been selected yet) — this is the exact list later passed into `build_store_from_uploads` (section 9).

**`st.text_input("Notify this email on upload", value=..., help=...)`** — renders a single-line text input box.
- The first argument is the label shown above/beside the box.
- `value=os.environ.get("SENDER_EMAIL", "")` — pre-fills the box with a default value: the sender's own email address (read from the environment, falling back to an empty string if not set) — a reasonable default assumption that most people testing this app will want to email themselves.
- `help="..."` — adds a small "?" tooltip icon next to the field; hovering over it shows this explanatory text.
- The return value, `notify_email`, is simply whatever string is currently in the box — updated live on every rerun as the user types (subject to Streamlit's usual rerun-on-interaction behavior).

---

## 24. Sidebar: the build button and its full handler

```python
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
```

This is the largest, most involved block in the sidebar — let's take it piece by piece.

**`if st.button("Build / Rebuild Knowledge Base", disabled=not uploaded_files):`** — renders a clickable button. `st.button(...)` returns `True` only on the specific script rerun that's directly caused by this exact button being clicked — on every other rerun (including reruns caused by *other* widgets), it returns `False`, so the whole indented block underneath simply doesn't execute. `disabled=not uploaded_files` grays out and disables the button entirely whenever the uploaded-files list is empty, preventing a pointless click with nothing to build from.

**`status_box = st.empty()`** — creates an empty placeholder UI element that can be updated/overwritten later in place, rather than appending new content below it every time. This is Streamlit's mechanism for showing *live-updating* content (like a progress message that changes over time) instead of a growing scroll of separate messages.

**`def show_progress(msg): status_box.info(msg)`** — defines a small local function, right here inside the button's `if` block. This is a **closure** — it "closes over" (remembers) the `status_box` variable from its enclosing scope, so calling `show_progress("some message")` later updates that specific placeholder with a blue info box containing that message. This is the exact `progress_callback` function that gets threaded all the way down through `build_store_from_uploads` → `build_vector_store_batched` → `_add_batch_with_retry` (sections 7-9), giving live, real-time feedback in the UI during a potentially slow embedding process.

**Initializing tracking variables before the `try`:**
```python
store, n_chunks, build_failed = None, 0, False
```
Setting sensible default values *before* attempting the risky operation, so that no matter what happens inside the `try`/`except` below, these variables are guaranteed to exist with sane values by the time the code afterward checks them.

**The try/except around the actual build:**
```python
try:
    store, n_chunks = build_store_from_uploads(uploaded_files, progress_callback=show_progress)
except Exception as e:
    build_failed = True
    status_box.empty()
    if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
        st.error(...)
    else:
        st.error(f"Failed to build the knowledge base: {e}")
```
Calls the full ingestion pipeline (section 9), passing in the `show_progress` closure so it can report live status. If *any* exception escapes all the way up from that call (recall section 8's retry logic already handles rate limits internally up to its retry limit — this `except` only catches what's left over after those retries are exhausted, or any entirely different kind of error), it's caught here: `build_failed` is set to `True` (a flag checked further down), the progress placeholder is cleared (`status_box.empty()` removes its content), and a specific, user-friendly error message is shown — again checking the error text for the rate-limit signature to give more targeted advice than a generic failure message.

**`status_box.empty()`** (outside the try/except, runs unconditionally) — clears the progress placeholder regardless of whether the build succeeded or failed, since by this point we're about to show a final success/error message instead.

**The three-way outcome check:**
```python
if build_failed:
    pass  # error already shown above
elif store is None:
    st.error("No readable text found in the uploaded file(s).")
else:
    ...
```
- `if build_failed:` with `pass` — an intentional no-op; if the exception handler already ran and displayed its own error message, there's nothing further to do here, but Python's syntax requires *some* statement inside an `if` block, so `pass` (which does literally nothing) satisfies that requirement.
- `elif store is None:` — this covers the *other* failure case from section 9: no exception was thrown, but `build_store_from_uploads` still legitimately returned `(None, 0)` because no extractable text was found in the uploaded PDF(s) (e.g. a scanned image-only PDF with no real text layer).
- `else:` — the success path, covered next.

**The success path:**
```python
st.session_state.store = store
st.session_state.messages = []
st.session_state.chat_history = []
st.success(f"Indexed {n_chunks} chunk(s) from {len(uploaded_files)} file(s).")
```
Saves the newly built store into persistent session state (so it survives future reruns), and **resets the chat** — clearing both history lists — since a freshly rebuilt knowledge base means old conversation context (which may reference the *previous* set of documents) is no longer meaningfully valid. `st.success(...)` shows a green confirmation box with the final chunk/file counts.

**The email notification:**
```python
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
```
- `if notify_email:` — only proceeds if the sidebar text field (section 23) isn't left blank.
- `", ".join(f.name for f in uploaded_files)` — a **generator expression** (`f.name for f in uploaded_files`, similar to a list comprehension but not materializing an intermediate list) producing each file's original name, immediately joined together with `", "` between them into one readable string like `"report.pdf, notes.pdf"`.
- The `body` string uses a multi-line, parenthesized string built from adjacent string literals (`f"..." f"..." f"..."` stacked on separate lines) — Python automatically concatenates adjacent string literals like this into one combined string, a common formatting technique for building longer text blocks readably.
- `st.spinner(f"Sending notification email...")` — another context manager, this one shows a temporary animated spinner with the given message for as long as the `with` block's code is executing, then automatically removes it once the block finishes.
- `send_email(notify_email, subject, body)` — this calls the **plain** `send_email` function directly (section 10), *not* the `@tool`-wrapped `email_this` — this particular email send is a deterministic, fixed action taken automatically by the app's own code whenever a build succeeds, not something the LLM is deciding to do. This is a useful contrast worth noting: not every use of `send_email` in this app goes through the LLM/tool-calling machinery — most of the time, it's just being called as an ordinary function.
- `if email_result.startswith("Email successfully sent"): st.success(...) else: st.warning(...)` — recall from section 10 that `send_email` always returns a descriptive string rather than raising an exception; this line inspects that string's *beginning* (`.startswith(...)`) to decide whether to show it as a green success box or a yellow warning box.

---

## 25. Sidebar: threshold slider

```python
    st.divider()

    threshold = st.slider(
        "Similarity threshold (lower = stricter)",
        min_value=0.3, max_value=1.5, value=DEFAULT_THRESHOLD, step=0.05,
        help="Chunks scoring above this distance are treated as not relevant."
    )
```

**`st.divider()`** — renders a simple horizontal line, purely a visual separator between sidebar sections.

**`st.slider(...)`** — renders a draggable slider widget. `min_value` / `max_value` set the allowed range, `value=DEFAULT_THRESHOLD` sets its starting position (0.8, from section 3), `step=0.05` controls the granularity of movement. The returned value, `threshold`, updates live as the user drags it and gets passed into `retrieve_filtered` (section 17) on every subsequent question — letting the user interactively tighten or loosen how strict retrieval filtering is, without touching any code.

---

## 26. Sidebar: New Chat button

```python
    st.divider()
    if st.button("🗑️ New Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
```

Another button, same pattern as section 24's build button (`True` only on the rerun where it was just clicked). Its handler is simple: reset both history lists back to empty — clearing the visible chat transcript *and* the condensing history — giving the user a clean slate to start a fresh conversation without needing to rebuild the knowledge base itself.

**`st.rerun()`** — explicitly forces an *immediate* script rerun, rather than waiting for the natural end of the current run. This ensures the UI updates right away to reflect the now-empty chat, rather than potentially showing stale content for a moment before the next natural interaction triggers a rerun.

---

## 27. Sidebar: hybrid agent toggle

```python
    st.divider()
    agent_mode = st.checkbox(
        "🤖 Enable email tool (hybrid agent)",
        value=False,
        help='Lets the model send an email when you explicitly ask, e.g. '
             '"email this summary to alex@gmail.com". Uses the same tool-calling '
             'pattern from the Day 1 email agent project.'
    )
```

**`st.checkbox(...)`** — renders a simple on/off checkbox. `value=False` means it's unchecked by default — so out of the box, the app behaves exactly as it did before this feature was added (section 20's `if not agent_mode:` branch), and a user has to deliberately opt in to enable tool-calling behavior. The returned boolean, `agent_mode`, is passed straight through to `answer_question` (section 31) on every chat turn.

---

## 28. Sidebar: chunk count caption

```python
    if st.session_state.store is not None:
        st.caption(f"Knowledge base: {st.session_state.store._collection.count()} chunks indexed")
```

A small, informational, grayed-out text line (`st.caption`, styled smaller/dimmer than normal text) shown only if a vector store currently exists in session state. `st.session_state.store._collection.count()` reaches into Chroma's underlying collection object (the same `._collection.count()` pattern seen back in section 6) to report the current total number of indexed chunks — a quick sanity-check/status indicator for the user.

---

## 29. Main page: title and empty state

```python
st.title("📄 RAG Document Assistant")
st.caption("Ask questions grounded in your own documents. Answers cite the source chunk they came from.")

if st.session_state.store is None:
    st.info("Upload one or more PDFs in the sidebar and click **Build / Rebuild Knowledge Base** to get started.")
else:
    ...
```

Note this code is **outside** the `with st.sidebar:` block (no indentation nesting it inside that context manager) — so `st.title` and everything following it renders in the main page area, not the sidebar.

**`st.title(...)`** — renders the large, page-level heading. **`st.caption(...)`** — a subtitle-like description directly underneath it.

**`if st.session_state.store is None:`** — checks whether a knowledge base has ever been successfully built (recall: `st.session_state.store` starts as whatever `load_existing_store()` returned at startup — section 21 — which is `None` on a completely fresh install with nothing built yet, or a real `Chroma` object once something has been indexed, whether from this session or a prior one).

**`st.info(...)`** — if there's no store yet, show a blue informational box guiding the user toward the sidebar's upload/build controls, rather than rendering an empty, confusing chat interface with nothing to actually query.

**`else:`** — if a store *does* exist, the entire chat interface renders instead (sections 30-31).

---

## 30. Main page: rendering past messages

```python
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for tag, label in msg["sources"].items():
                        st.markdown(f"**{tag}** — {label}")
```

This loop **redraws the entire chat history** on every script rerun. Remember: Streamlit reruns the whole script from top to bottom constantly (section 4) — there's no persistent, incrementally-updated DOM the way a typical JavaScript chat app might maintain. Instead, every single rerun, this loop walks through the *entire* `st.session_state.messages` list (which itself persists correctly across reruns, per section 21) and redraws every single past message from scratch.

**`for msg in st.session_state.messages:`** — iterates over the list of message dictionaries built up over the conversation.

**`with st.chat_message(msg["role"]):`** — a Streamlit component specifically designed for chat interfaces; it renders a styled message bubble with an appropriate avatar icon based on the `role` string (`"user"` or `"assistant"`), and everything inside this `with` block appears visually grouped inside that bubble.

**`st.markdown(msg["content"])`** — renders the message's text content, interpreting it as Markdown (so things like `**bold**` or bullet points in the model's response render with proper formatting rather than showing the raw asterisks).

**`if msg.get("sources"):`** — recall from section 21 that a message dictionary's `"sources"` key is either a citation-map dictionary (for assistant messages that had grounded context) or `None` (for user messages, or assistant messages with no relevant context found). `msg.get("sources")` safely returns that value (or `None` if the key is somehow missing entirely), and both an empty dictionary `{}` and `None` are "falsy" in Python, so this check naturally skips rendering a sources section when there's nothing to show.

**`with st.expander("Sources"):`** — a collapsible section, closed by default, labeled "Sources" — clicking it reveals its contents. This keeps the main chat readable (citations aren't dumped inline by default) while still making them available on demand.

**`for tag, label in msg["sources"].items():`** — `.items()` on a dictionary yields `(key, value)` pairs; here that's `(tag, label)`, e.g. `("Source 1", "report.pdf (page 3)")`. **`st.markdown(f"**{tag}** — {label}")`** renders each one as a bolded tag followed by its description.

---

## 31. Main page: handling a new chat turn

```python
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
                    st.session_state.chat_history, threshold, agent_mode
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
```

**`user_input = st.chat_input("Ask a question about your documents...")`** — renders the text input box permanently pinned to the bottom of the page (a purpose-built Streamlit component for chat interfaces), with the given placeholder text shown when empty. Its return value is `None` on every rerun *except* the specific one triggered by the user pressing Enter/submitting a message — on that rerun, it returns the actual text they typed.

**`if user_input:`** — since `None` and an empty string are both falsy, this block only runs when the user has actually just submitted a real message.

**Recording and displaying the user's message:**
```python
st.session_state.messages.append({"role": "user", "content": user_input, "sources": None})
with st.chat_message("user"):
    st.markdown(user_input)
```
First, the new message is appended to the persistent history list (so it'll be included in future reruns' redraw loop, section 30). Then it's *also* rendered immediately here — this immediate render is necessary because the current script run needs to show the user's just-typed message right away, without waiting for a subsequent rerun; the history loop (section 30) already ran *earlier* in this same script execution, before this new message existed, so it wouldn't have included it.

**Generating and displaying the assistant's response:**
```python
with st.chat_message("assistant"):
    with st.spinner("Thinking..."):
        llm = get_llm()
        answer, citation_map, standalone = answer_question(
            st.session_state.store, llm, user_input,
            st.session_state.chat_history, threshold, agent_mode
        )
        st.markdown(answer)
        if citation_map:
            with st.expander("Sources"):
                for tag, label in citation_map.items():
                    st.markdown(f"**{tag}** — {label}")
        if standalone != user_input:
            st.caption(f"(interpreted as: \"{standalone}\")")
```
- `with st.chat_message("assistant"):` — opens an assistant-styled chat bubble.
- `with st.spinner("Thinking..."):` — shows an animated spinner with this label for the duration of whatever code runs inside the block — here, that covers the entire (potentially several-second) `answer_question` call.
- `llm = get_llm()` — grabs the cached plain LLM instance (section 4). Note this is passed into `answer_question` specifically for use in the *condensing* step (section 16) — the function internally decides whether to also use the separately-cached tool-bound model (section 12), based on the `agent_mode` argument.
- `answer_question(st.session_state.store, llm, user_input, st.session_state.chat_history, threshold, agent_mode)` — this is the single call that runs the *entire* pipeline described in section 20: condense → retrieve → filter → format → generate (in either plain or hybrid-agent mode) — passing in the live vector store, the raw user input, the existing condensing-history list, the current slider threshold value, and the current checkbox state.
- `st.markdown(answer)` — displays the final answer text.
- `if citation_map:` / the `st.expander` block — same citation-rendering pattern as section 30, shown for this brand-new response.
- `if standalone != user_input:` — compares the original text the user typed against the (possibly rewritten) standalone version that was actually used for retrieval. If the condensing step actually changed anything (meaning `standalone` differs from `user_input`), a small caption is shown revealing what the question was interpreted as — giving the user transparency into that "hidden" rewriting step (section 13) rather than leaving it invisible.

**Updating both history lists after the turn completes:**
```python
st.session_state.messages.append({"role": "assistant", "content": answer, "sources": citation_map})
st.session_state.chat_history.append(HumanMessage(content=user_input))
st.session_state.chat_history.append(AIMessage(content=answer))
```
Three final appends: the assistant's message (with its citation map) goes into the UI-facing `messages` list; and — importantly — the **original** `user_input` (not the condensed `standalone` version) gets wrapped in a `HumanMessage` and added to `chat_history`, alongside an `AIMessage` wrapping the final answer. Using the original user phrasing here (rather than the internally-rewritten standalone question) keeps the stored conversation transcript reading the way the exchange actually happened from the user's perspective, even though the *retrieval* step internally used the rewritten version just for that one lookup.

---

## 32. Putting it all together — the full request lifecycle

To close out, here's the entire flow traced end-to-end, from a user typing a question to seeing an answer on screen — referencing every section above by number:

1. **App starts** (§21): Streamlit runs the whole script. Cached resources (§4, §12) are created once. Session state initializes — if a knowledge base already exists on disk from a previous session, it's silently reloaded (§6) into `st.session_state.store`.
2. **User uploads PDFs and clicks "Build / Rebuild Knowledge Base"** (§22-24): files are saved to disk, loaded, split into chunks, embedded in rate-limit-safe batches (§7-9) into a freshly-named Chroma collection, and the store is saved into session state. An email notification optionally fires (§10, §24).
3. **User types a question and submits it** (§31): the raw text is immediately recorded and displayed.
4. **`answer_question` runs** (§20):
   - The question is condensed against prior conversation history if any exists (§13, §15, §16).
   - The (possibly rewritten) standalone question is used to run a filtered similarity search against the vector store (§17).
   - Retrieved chunks (or the "nothing relevant" sentinel) get formatted into a citation-labeled context string (§18).
   - **If agent mode is off** (§14, §20 branch 1): a single LCEL chain fills the grounded-answer prompt and generates a response.
   - **If agent mode is on** (§19, §20 branch 2): the tool-bound model decides whether to just answer, or to call `email_this` (§11) — which, if invoked, actually sends a real email (§10) — before producing a final natural-language response.
5. **The answer is displayed** (§31), along with its sources (if any) and, if applicable, a note showing how an ambiguous question was interpreted.
6. **Both history lists get updated** (§31) so the next turn has full context — and the whole cycle repeats on the next message, with Streamlit rerunning the entire script from the top every time (§4, §21), relying entirely on `st.session_state` to make it *feel* like a continuous, persistent conversation despite that constant re-execution under the hood.

---

*This document is meant to be read alongside the actual `app.py` file open side-by-side — every section title maps directly to a labeled block of that file.*
