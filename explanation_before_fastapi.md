# `app.py` — The Super Simple Walkthrough

Same code, same order, same sections as before — but this time every explanation is written like I'm explaining it to a friend over coffee, using real-life comparisons instead of jargon.

**The big picture analogy we'll use throughout:**
> Imagine you hire a **smart librarian** for your office. You hand them a stack of PDFs. The librarian reads every page, memorizes where every fact lives, and from then on, whenever you ask a question, they instantly flip to the *exact* page that has your answer and read it to you — instead of making things up. If you ask them to email someone, they'll only do it if you clearly tell them to — otherwise they just answer your question and mind their business.

That librarian is this app. Let's meet every part of how they work.

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

**Simple version:** This is just the **cover page** of the file. Like sticking a note on the front of a folder that says "This is the librarian project — built over 7 days, here's what each day added." It doesn't *do* anything when the code runs — it's purely there so a human (you, your teammate, future-you) can understand what this file is about at a glance, without reading a single line of actual code.

The last line — "Run with: `streamlit run app.py`" — is like an instruction manual sticker: *don't try to start this the normal Python way (`python app.py`)*, use the special Streamlit launcher instead, because Streamlit needs to set up a live webpage for you.

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

**Simple version:** Think of this as **gathering your tools before starting a job** — like a chef laying out knives, pans, and ingredients before cooking. Each import is one tool for one job:

| Tool | Real-life comparison |
|---|---|
| `os`, `re`, `time` | Your basic toolbox — a ruler, a pair of scissors, a stopwatch. Nothing fancy, just everyday utilities. |
| `smtplib`, `email.mime...` | The **mailbox and envelope kit** — lets you actually stuff a letter into an envelope and mail it. |
| `streamlit` | The **whiteboard and markers** — this is what draws the actual webpage you see: buttons, boxes, chat bubbles. |
| `dotenv.load_dotenv` | A **locked drawer key** — lets the app quietly grab secret passwords from a hidden file instead of writing them out in the open. |
| `ChatGoogleGenerativeAI`, `GoogleGenerativeAIEmbeddings` | Two different **hired brains**: one is the actual "librarian" who reads and talks; the other is a translator whose only job is turning sentences into a special numeric "fingerprint." |
| `Chroma` | The **filing cabinet** where those numeric fingerprints get stored so they can be searched later. |
| `PyPDFLoader` | The **photocopier** — feeds in a PDF, spits out readable pages. |
| `RecursiveCharacterTextSplitter` | The **paper cutter** — slices big pages into small, bite-sized note cards. |
| `ChatPromptTemplate` | A **fill-in-the-blanks form letter** — same wording every time, just swap in the details. |
| `StrOutputParser` | A **strainer** — the librarian's answer comes back wrapped in extra packaging; this strips it down to just the plain words. |
| `HumanMessage`, `AIMessage` | **Name tags** — stuck onto each message so you can tell "this was said by the user" vs. "this was said by the assistant." |
| `tool` | A **badge-maker** — turns an ordinary function into something the AI is allowed to "request," like giving the librarian's assistant permission to use the mail machine. |

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

**Simple version:**

- **`load_dotenv()`** — imagine you have a secret notebook (`.env`) with your passwords written in it, hidden in a drawer. This line opens that notebook and quietly copies its contents into the app's short-term memory, so the app can use those passwords without you ever typing them into the visible code.

- **`GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")`** — this is literally just: *"Go check that notebook for the entry called GEMINI_API_KEY, and keep it handy."* If it's not there, you just get nothing (`None`) instead of a crash — like checking a coat hook and finding it empty rather than the wall falling down.

- **`DATA_DIR = "data"` / `PERSIST_DIR = "chroma_db"`** — these are just **labels for two folders/boxes**: one box (`data`) is where you dump the raw PDFs you upload. The other box (`chroma_db`) is the filing cabinet where the "fingerprints" (embeddings) of those PDFs get stored. Giving them names once means the rest of the code can just say "put it in `DATA_DIR`" instead of re-typing "data" everywhere and risking a typo.

- **The big comment above `ACTIVE_COLLECTION_FILE`** — here's the real-life story: imagine every time you get new documents, instead of throwing out the old filing cabinet and buying a new one (which sometimes jams the drawer and won't open — that's the Windows crash the comment describes), you just **buy a brand new small filing cabinet and put a sticky note on your desk saying "use THIS cabinet now."** Nothing ever gets thrown away or force-removed; you just point to a different cabinet. Simple, no jams, no fuss.

- **`ACTIVE_COLLECTION_FILE`** — that's literally the **sticky note itself** — a small file that just says which cabinet (which batch of indexed documents) is the current one to use.

- **`DEFAULT_THRESHOLD = 0.8`** — think of this as a **"how picky is the librarian"** dial. If someone asks a question, how close does a page have to match before the librarian is willing to say "yes, this is relevant"? `0.8` is the starting pickiness level.

- **`st.set_page_config(...)`** — this is just **decorating the browser tab**: giving it a title ("RAG Document Assistant"), a little icon (📄), and telling it to use the *whole* width of the screen instead of a squished narrow column — like choosing a wide-screen TV over a small square one.

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

**Simple version — the weirdest thing about Streamlit first:**

Imagine every time you click *anything* on this webpage — even just moving a slider — the **entire script re-runs from the very top, like restarting the whole program from scratch**, every single time. That's genuinely how Streamlit works.

Now imagine if, every time that happened, the app had to **re-hire the librarian and re-hire the translator from zero** — like firing and rehiring the same staff member every time someone opens a door in the office. That would be exhausting and slow.

**`@st.cache_resource`** is the fix: it's like putting a sign on the librarian's desk that says *"Already hired — don't rehire, just reuse the same person."* The very first time the function runs, it actually does the work (hires the librarian). Every time after that — even after the whole script "restarts" — it just points back to the same already-hired person instead of doing the hiring process again.

`show_spinner=False` just means: *don't bother showing a "hiring in progress..." loading icon* — hiring is basically instant here, so there's no point flashing a spinner.

**Two functions, two different staff members:**
- `get_embeddings_model()` hires the **translator** — their only job is turning sentences into number-fingerprints. They don't chat, they don't answer questions, they just translate text into numbers.
- `get_llm()` hires the **librarian** — the one who actually reads, thinks, and talks back to you.

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

**Simple version:** Remember the sticky note idea from Section 3? These two functions are literally **"read the sticky note"** and **"write a new sticky note."**

- **`_read_active_collection()`** — walks over to the desk, checks if there's a sticky note at all. If yes, reads what's written on it (the name of the current filing cabinet) and hands that back. If there's no sticky note yet (brand new app, nothing built), it just shrugs and says "nothing here" (`None`).

- **`_write_active_collection(name)`** — grabs a fresh sticky note, writes the new cabinet's name on it, and sticks it on the desk — replacing whatever was there before. It also double-checks the desk itself (`chroma_db` folder) exists first, just in case this is the very first time anything's been built.

The little underscore `_` at the start of both names is just office slang for *"internal use only — not meant for outsiders to touch directly."* It's a hint to other developers, not an actual lock.

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

**Simple version:** This is what happens the moment you **walk back into the office the next morning.** Before doing anything else, you check: *"Did I leave a filing cabinet set up yesterday?"*

1. `collection_name = _read_active_collection()` — check the sticky note (Section 5). Maybe it says `"docs_1732481920"`, maybe there's no note at all.
2. `if collection_name and os.path.isdir(PERSIST_DIR):` — only bother continuing if there **is** a note *and* the filing room (`chroma_db` folder) actually exists. No point checking a cabinet that was never built.
3. `Chroma(persist_directory=..., ...)` — walk over and **open that specific cabinet drawer**. This doesn't create anything new or copy anything — it just opens a handle to what's already sitting there.
4. `store._collection.count() > 0` — peek inside: is the drawer actually got *anything* in it, or is it empty/broken? A basic sanity check before trusting it.
5. If everything checks out: hand back the open cabinet, ready to use. If not: shrug, return "nothing to use" (`None`), and the app will just start fresh.

**Why this matters in real life:** without this, you'd have to **re-upload and re-index all your PDFs every single time you restarted the app** — even if nothing changed. This function is what lets your work survive overnight.

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

**Simple version — real-life comparison:** Imagine you're calling a customer service line and it's busy. Sometimes the automated message actually tells you: *"Please call back in 43 seconds."* Wouldn't it be silly to ignore that and just guess a random wait time instead?

That's exactly what this function does. Google's translator service (the embedding model) has a speed limit — like a "only 100 calls per minute" rule at a call center. If you go over that limit, instead of just saying "busy, try later," Google's error message actually often says something like *"Please retry in 43.4s"* — buried inside a big scary error text.

This function is like a person with really good eyesight, **scanning that scary error message specifically for the phrase "retry in ___s"**, plucking out just the number, and handing it back so the app knows exactly how long to wait — instead of guessing. It even adds **1 extra second as a small safety cushion**, just in case.

If that specific phrase isn't found in the error (maybe it's a totally different kind of problem), it just says "couldn't find a hint" (`None`), and something else will pick a default wait time instead.

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

**Simple version — think of a busy photocopier at the office:**

You bring a stack of papers (a "batch" of document chunks) to the office photocopier (the translator/embedding service) to get them copied. Sometimes the copier says: *"Sorry, too busy right now, come back in a bit."*

This function is the **patient coworker standing at the copier** who handles that:

1. **Try to copy the batch** (`store.add_documents(batch)`). If it works — great, done, walk away (`return`).
2. **If the copier complains** (an error happens): check *what kind* of complaint it is.
   - Is it specifically the **"too busy"** complaint (`RESOURCE_EXHAUSTED` or `429`)? If **not** — if it's some totally different problem, like "paper jam" or "out of toner" — there's no point waiting around; immediately give up and pass the problem along (`raise`).
   - Also, if you've **already tried the maximum allowed number of times** (2 tries total), stop waiting and give up too — no point standing there forever.
3. **If it genuinely was just "too busy," and you've still got tries left:** figure out how long to wait (using the "retry in ___s" hint from Section 7, or defaulting to 30 seconds, but never waiting more than 65 seconds no matter what).
4. **Tell whoever's watching** ("Rate limit hit, waiting 43s...") so the person isn't left staring at a frozen screen wondering what's happening.
5. **Actually wait** that many seconds (`time.sleep(wait)`), then loop back and try again.
6. If somehow you get through all attempts without success or a clean failure, throw one final, clear error explaining what happened.

**Real-life takeaway:** this is a "**try again politely, but don't wait forever**" strategy — very similar to how you might redial a busy phone line a couple of times, but eventually give up and try a different approach instead of holding forever.

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

**Simple version — imagine you have 320 index cards to laminate**, but the laminating machine can only handle a stack of 50 at a time without overheating.

1. **Set up a brand new empty filing cabinet drawer** (`Chroma(...)`) — nothing's in it yet.
2. **Count your total cards** (`total = len(chunks)`).
3. **Grab the first 50 cards, laminate them, put them away, grab the next 50, laminate, put away... repeat** — that's exactly what `for i in range(0, total, batch_size): batch = chunks[i:i+batch_size]` is doing.
4. **Announce progress out loud** each time ("Laminating cards 1–50 of 320...") so whoever's watching knows it's working, not frozen.
5. **Laminate this batch, using the patient-coworker retry trick from Section 8** in case the machine says "too busy."
6. **Take a short breather (2 seconds) between batches** — unless this was the very last batch, in which case there's no need to pause after finishing. This breather is a *preventative* measure — spacing things out so you're less likely to hit the "too busy" wall in the first place.
7. **Once every batch is done, hand back the full drawer**, now completely filled with laminated cards.

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

**Simple version — this is the entire "new employee onboarding" process for your documents, start to finish:**

1. **Make sure there's a physical inbox tray on the desk** (`os.makedirs(DATA_DIR, exist_ok=True)`) — if it's not there, create it.

2. **Physically print out and file every uploaded PDF into that tray:**
   ```python
   for uf in uploaded_files:
       path = os.path.join(DATA_DIR, uf.name)
       with open(path, "wb") as f:
           f.write(uf.getbuffer())
       saved_paths.append(path)
   ```
   Remember, when you upload a file in your browser, it's just sitting in the browser's memory — not actually saved anywhere permanent yet. This loop is like **taking each uploaded file out of your hands and physically placing a printed copy into the inbox tray on disk**, one at a time, keeping a checklist (`saved_paths`) of exactly what got filed.

   > **Important real-life detail:** this only files what you *just* handed over — it deliberately ignores any old papers that might already be sitting in that tray from a previous day. So even if the tray has leftover clutter from before, only today's fresh uploads get processed.

3. **Read every page of every filed document:**
   ```python
   documents = []
   for path in saved_paths:
       documents.extend(PyPDFLoader(path).load())
   ```
   This is the **photocopier scanning every single page** of every PDF you just filed, turning each page into a readable digital note (with a sticky label saying which file and page number it came from).

4. **Check: did we actually get any readable text at all?**
   ```python
   if not documents:
       return None, 0
   ```
   If every PDF turned out to be blank, corrupted, or was just a scanned image with no real text (like a photo of a page, not actual typed text) — there's simply nothing to work with, so stop here and say "nothing built."

5. **Cut everything into bite-sized note cards:**
   ```python
   splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
   chunks = splitter.split_documents(documents)
   ```
   A full page is often too much text to search through efficiently — like handing someone an entire chapter when they just wanted one paragraph. So this **slices each page into roughly 1000-character note cards**, and — cleverly — lets consecutive cards **share a little overlap (200 characters)**, like photocopying the last sentence of one card onto the start of the next, so you never lose a thought that happened to land right at a cut point.

6. **Label this batch with today's exact timestamp so it's unique:**
   ```python
   collection_name = f"docs_{int(time.time())}"
   ```
   This is like **writing today's exact time on a new filing cabinet drawer** — `"docs_1732481920"` — guaranteeing it's never confused with any previous drawer.

7. **Laminate and file everything (Section 9's first function) into that fresh drawer.**

8. **Update the sticky note on the desk** (`_write_active_collection`) to say "use THIS drawer now."

9. **Report back:** hand over the finished drawer and how many note cards were made — useful for showing the user a friendly "Indexed 320 cards!" message.

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

**Simple version:** This is just the app's **personal mail clerk**, who knows how to physically walk to the post office and mail a letter. Nothing about AI here at all — it's a plain, boring, reliable utility.

- **Grab the return address and the "special access key"** (`sender_email`, `app_password`) — think of `app_password` like a **special guest pass** rather than your actual house key. Gmail requires this special pass for programs (rather than humans typing in a browser) to be allowed to send mail — it's safer, because you can cancel just the guest pass without changing your main password.
- **Write the letter:**
  ```python
  msg = MIMEMultipart()
  msg['From'] = sender_email
  msg['To'] = recipient_email
  msg['Subject'] = subject
  msg.attach(MIMEText(body, 'plain'))
  ```
  Literally filling out an envelope: who it's from, who it's to, the subject line, and stuffing the actual letter (`body`) inside.
- **Walk to the post office and hand it over:**
  ```python
  with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
      server.login(sender_email, app_password)
      server.sendmail(sender_email, recipient_email, msg.as_string())
  ```
  Opens a secure line to Gmail's post office, shows the guest pass to get in, and hands the letter over to be delivered.
- **Report back what happened, no matter what:**
  Instead of the clerk just collapsing and refusing to talk if something goes wrong, they always **come back and tell you clearly what happened** — "delivered successfully," or "sorry, your guest pass was rejected," or "sorry, that address doesn't exist," or a general "something else went wrong." This matters a lot because later, an AI will be reading this clerk's report to decide what to tell the user — so the clerk needs to *always* say something sensible, never just vanish.

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

**Simple version:** Remember the mail clerk from Section 10? Right now, only *your code* knows the clerk exists — the AI librarian has no idea they can ask the clerk to do anything.

`@tool` is like **giving the librarian a laminated instruction card** that says: *"Psst — there's a mail clerk on staff. Here's their name (`email_this`), here's exactly what to hand them (a recipient, a subject, a message), and here's when you're allowed to use them."*

The **most important part** is the instructions written on that card (the docstring) — because the librarian (the AI) **never reads the clerk's actual paperwork/code**. All the librarian ever sees is:
- The clerk's name (`email_this`)
- The instruction card's wording — which is written like a strict rule for the librarian to follow: *"Use this ONLY when the user explicitly asks... Never use this tool unless the user clearly asked."*
- What info to hand over (recipient, subject, message)

That's why this docstring reads more like a **rulebook for the AI** than a description for a human developer — because in this one case, the AI genuinely is the "reader" of this text.

The function body itself, `return send_email(...)`, just means: *when the librarian asks for this, quietly forward the request straight to the real mail clerk from Section 10.*

---

## 12. A second, tool-bound LLM instance

```python
@st.cache_resource(show_spinner=False)
def get_llm_with_tools():
    """Same Gemini model as get_llm(), but with the email tool bound — this is the
    Day 1 tool-calling pattern (bind_tools) applied inside the RAG app."""
    return get_llm().bind_tools([email_this])
```

**Simple version:** Imagine you have **the same librarian**, but sometimes you hand them the laminated instruction card about the mail clerk (Section 11), and sometimes you deliberately **don't** — because you don't always want them to have the *option* of mailing things.

`get_llm().bind_tools([email_this])` is exactly that: **take the same librarian, but this time, additionally hand them the instruction card about the mail clerk.** The result is basically a second version of the same librarian — one who *knows about* the mail clerk and can request their help, versus the original who has no idea the clerk even exists.

**Why bother keeping two separate versions instead of just always giving everyone the card?** Because there's a toggle switch in the app (Section 27) letting the user decide: *"Should the librarian even be allowed to consider mailing things today?"* Keeping two clearly separate librarian-versions means there's zero chance of accidentally letting the librarian mail something when the user never wanted that option available in the first place.

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

**Simple version — real-life scenario:** Imagine you ask a librarian, *"What's the refund policy?"* They answer. Then you follow up with, *"What about after that?"*

If the librarian tried to search their filing cabinet for the literal words *"what about after that"* — they'd find nothing useful, because that sentence, on its own, means nothing. It only makes sense **because of the conversation that came before it.**

This is a **fill-in-the-blank instruction card** the librarian reads *before* searching, that basically says: *"Take whatever the person just said, and if it's a vague follow-up that only makes sense because of earlier conversation, rewrite it into a complete, standalone sentence that would make sense to a total stranger who wasn't listening to the earlier conversation."*

So "What about after that?" becomes something like "What is the refund policy after 30 days?" — a fully self-contained question — **before** any searching happens.

The line *"Do not answer the question - only rewrite it"* is there because otherwise the librarian might get eager and just try to answer right then and there — but at this exact stage, all we want is the *rewritten question*, not an actual answer yet.

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

**Simple version:** This is the **rulebook you hand the librarian right before they answer a question**, and every rule solves a specific real-world problem:

- **"using ONLY the context provided below" / "Do not use outside knowledge."** — This is like telling the librarian: *"Only tell them what's written on these specific pages I just handed you. Even if you personally happen to know the answer from your own general knowledge, don't use that — only use these pages."* Without this rule, the librarian might "helpfully" mix in stuff they remember from elsewhere, which defeats the whole point of trusting *your* documents specifically.
- **"Reference the [Source N] tags..."** — *"When you tell them something, point to exactly which page you got it from,"* like a librarian saying "According to page 4 of the handbook..." instead of just stating facts with no proof.
- **The `NO_RELEVANT_CONTEXT` rule** — *"If I didn't actually hand you any relevant pages this time, don't make something up — just honestly say you don't have the answer."* This stops the librarian from guessing when they genuinely have nothing to go on.
- **"You may refer naturally to earlier parts of the conversation, but never invent facts..."** — *"You can still be a normal, friendly conversationalist and reference things said earlier — just don't invent new *facts* while doing it."* Being chatty and being accurate aren't the same thing, and this rule keeps both intact separately.
- **"Be concise and direct."** — *"Don't ramble — get to the point."*

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

**Simple version:** Imagine the whole conversation so far is stored as a stack of labeled index cards — some tagged "User said this," some tagged "Assistant said this." But the fill-in-the-blank form from Section 13 just wants **one plain paragraph of readable text**, not a stack of index cards.

This function is the **person who takes that stack of cards and retypes it into a clean, readable transcript**, like:
```
User: What's the refund policy?
Assistant: Refunds are allowed within 30 days.
```

- **If there's no conversation yet at all** — just write "(no previous conversation)" instead of leaving it blank and confusing.
- **Otherwise**, go through every card one at a time, figure out if it was said by the "User" or the "Assistant" (checking its label), write it out as `"Label: what they said"`, and stack all those lines into one clean paragraph, one sentence per line.

---

## 16. The condensing function

```python
def condense_question(question, chat_history, llm):
    if not chat_history:
        return question
    chain = CONDENSE_PROMPT | llm | StrOutputParser()
    return chain.invoke({"chat_history": format_history(chat_history), "question": question}).strip()
```

**Simple version:** This is the actual **"send the vague follow-up question to the librarian and get back a clear, standalone version."**

- **`if not chat_history: return question`** — Real-life shortcut: if this is literally the very first thing anyone's ever said in this conversation, there's *nothing to be vague about* yet — no earlier context to be confused by. So just skip the whole rewriting step entirely and use the question exactly as typed. This also saves you from bothering the librarian with an unnecessary extra question when it's not needed.

- **`chain = CONDENSE_PROMPT | llm | StrOutputParser()`** — think of this as **an assembly line with three stations**: Station 1 fills out the form letter from Section 13. Station 2 hands that filled-out form to the librarian to think about. Station 3 takes the librarian's reply and strips away any extra wrapping, leaving just the plain rewritten sentence. The `|` symbol is just "conveyor belt to the next station."

- **`chain.invoke({...})`** — actually **push the conveyor belt into motion**, feeding in the real conversation history and the real question.

- **`.strip()`** — tidy up any stray blank spaces the librarian might've left at the start/end of their reply, like trimming the edges off a photocopy.

---

## 17. Filtered retrieval

```python
def retrieve_filtered(store, query, k=5, threshold=DEFAULT_THRESHOLD):
    results = store.similarity_search_with_score(query, k=k)
    return [(doc, score) for doc, score in results if score <= threshold]
```

**Simple version — real-life comparison:** Imagine you ask the librarian a question, and they go through the filing cabinet and pull out **the 5 pages that seem closest to your question** — that's just what `similarity_search_with_score` does. But here's the catch: **the cabinet will always hand back 5 pages, even if none of them are actually relevant** — it's like a search engine that insists on showing you 5 results even if your search term matched nothing well; it just shows you the "least bad" options.

So the second line — the filtering step — is the librarian **actually looking at those 5 pages critically and asking, "Wait, is this page *actually close enough* to be useful, or is it just the best of a bad bunch?"** Any page that's too far off gets tossed aside, even if it technically made the "top 5" list.

**Lower score = closer match** (a bit counterintuitive — think of it as a "distance," like how many steps away something is, not a percentage of similarity). So `score <= threshold` means: *"keep only the pages that are close enough — within this many steps — to trust."*

**Why this matters in real life:** without this filtering step, ask the librarian something totally unrelated to your documents (like "what's the weather today?") and they'd *still* be handed 5 random pages and might awkwardly try to force an answer out of them, instead of honestly saying "I don't have anything relevant for that."

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

**Simple version:** Imagine the librarian just pulled out a small stack of genuinely relevant pages. Before handing them to the "brain" that writes the actual answer, someone needs to **staple a little numbered sticky tab onto each page** ("Source 1," "Source 2"...) so that later, whoever reads the final answer can trace exactly which page each fact came from — like footnotes in an essay.

- **If nothing relevant was found at all** — just hand back a special "nothing here" marker (`"NO_RELEVANT_CONTEXT"`) and an empty tab list. This is the exact marker that the rulebook in Section 14 is watching for.

- **Otherwise, for each relevant page:**
  - Slap on a numbered tab: "Source 1", "Source 2", etc.
  - Note down which actual file and page number it came from (with safe fallback labels like "unknown"/"?" if that info is somehow missing).
  - Strip the file path down to just the filename — nobody needs to see the whole computer folder structure, just "handbook.pdf" is enough.
  - Stack the tabbed page's actual text into a big combined stack, with the tab label written right above each page's text.

- **Finally, staple everything into one combined packet** ("\n\n".join...) and hand back both: the full readable packet, *and* a little lookup chart mapping "Source 1" → "handbook.pdf (page 4)" for later reference.

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

**Simple version:** This is the **exact same rulebook as Section 14**, copy-pasted almost word for word — **plus one brand-new rule bolted on at the end**, specifically about the mail clerk:

> *"You have access to an 'email_this' tool. Use it ONLY if the user explicitly asks you to email, send, or mail something... do not send an email unless clearly asked to."*

**Real-life comparison:** think of this like handing the librarian a rulebook that mostly says the same things as before ("stick to the facts, cite your sources, don't guess"), but with **one extra warning sticky note slapped on top**, since *this* version of the librarian also has access to the mail clerk: *"Yes, you can call the mail clerk — but ONLY if the customer clearly and explicitly asks you to mail something. Don't get trigger-happy with the mail clerk on your own initiative."*

Why repeat this warning here **and** also on the mail clerk's own instruction card back in Section 11? Because when there's a real-world action with real consequences (an actual email genuinely gets sent to someone), **it's worth saying the same important safety rule twice, in two different places**, rather than risking the librarian missing it if it only appeared once. It's the same principle as a pilot's pre-flight checklist repeating critical safety checks even when they feel redundant — redundancy is a *feature*, not a mistake, when the stakes matter.

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

**Simple version:** This is the **whole front-desk process, start to finish, every time a customer asks a question** — this one function is basically "the manager" who runs through every step in order.

**The first three steps happen no matter what** (whether the mail clerk feature is turned on or not):
```python
standalone = condense_question(question, chat_history, llm)
filtered = retrieve_filtered(store, standalone, threshold=threshold)
context, citation_map = format_docs_with_citations(filtered)
```
1. **Clean up the question** if it's vague, using earlier conversation (Section 16).
2. **Go dig through the filing cabinet**, keeping only genuinely relevant pages (Section 17).
3. **Staple numbered sticky tabs** onto whatever pages were found (Section 18).

Now the manager checks a switch on the wall: *"Is 'agent mode' turned on today?"*

**If it's OFF (the normal, default day at the office):**
```python
chain = ANSWER_PROMPT | llm | StrOutputParser()
answer = chain.invoke({"context": context, "question": standalone})
return answer, citation_map, standalone
```
Simple assembly line: fill out the rulebook form with the tabbed pages and question → hand to the librarian → strip the wrapping off their reply → done. Hand back the answer, the sticky-tab chart, and the cleaned-up question.

**If it's ON (the special "mail clerk available" day):**

```python
llm_tools = get_llm_with_tools()
prompt_text = HYBRID_INSTRUCTIONS.format(context=context, question=standalone)
messages = [HumanMessage(content=prompt_text)]
```
Fetch the *version of the librarian who knows about the mail clerk* (Section 12), fill out the special rulebook (Section 19), and start a running conversation log with that first message in it.

```python
response = llm_tools.invoke(messages)
messages.append(response)
```
**Ask the librarian to respond.** Whatever they say — whether it's a plain answer, or a request like "I'd like to call the mail clerk" — jot it down in the running log too.

```python
if response.tool_calls:
```
Check: **did the librarian actually ask to use the mail clerk this time**, or did they just answer normally in words?

**If they DID ask for the mail clerk:**
```python
for tool_call in response.tool_calls:
    tool_result = email_this.invoke(tool_call)
    messages.append(tool_result)
final = llm_tools.invoke(messages)
answer = final.content
```
This is the real moment the **actual email gets sent** — the mail clerk (Section 11 → 10) genuinely walks to the post office and delivers a real message. Whatever the clerk reports back ("delivered!" or "failed because...") gets jotted into the log too. Then, **the librarian is asked once more** — now that they can see what the clerk reported — so they can give the customer a clean, human sentence like *"I've sent that summary to alex@gmail.com!"* instead of just dumping the clerk's raw internal report on the customer.

**If they did NOT ask for the mail clerk** (the normal case for most questions):
```python
else:
    answer = response.content
```
Just use whatever they said directly — no mail clerk was needed.

**Either way, hand back the same three things** (answer, sticky-tab chart, cleaned-up question) — so whoever called this function doesn't even need to know or care which path was taken behind the scenes.

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

**Simple version:** Remember from Section 4 — **every single click on the page basically restarts the whole script from scratch.** Imagine if, every time someone in the office moved a chair, the *entire office's memory got wiped* and everyone forgot everything that happened five seconds ago. That would make having a conversation completely impossible — you'd forget what was just said before you even finished replying!

**`st.session_state`** is like a **special notebook that survives the memory-wipe** — even though everything else resets, this notebook stays exactly as it was. It's the app's genuine long-term memory *for this specific visitor's browser tab.*

**The pattern `if "key" not in st.session_state: ...`** is basically: *"Only write a brand-new blank page in the notebook if there isn't already a page with this name. If a page already exists, leave it completely alone — don't erase what's already written on it."* This check is what prevents the chat from being wiped blank on every single click.

**Three separate notebook pages, each for a different job:**
- `st.session_state.messages` — the page that stores **exactly what gets shown on screen** in the chat window (who said what, and any source citations attached).
- `st.session_state.chat_history` — a separate, more minimal page that only stores what's needed for the "clean up vague questions" step (Section 16) — it doesn't need the extra display details the other page has.
- `st.session_state.store` — the actual open filing cabinet drawer. Starts by checking if there's already one sitting around from before (Section 6), so you don't lose your indexed documents just because the browser tab reloaded.

---

## 22. Sidebar: header and API key check

```python
with st.sidebar:
    st.header("📄 Knowledge Base")

    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY not found. Add it to your .env file.")
```

**Simple version:** `with st.sidebar:` is like saying, *"Everything I build from now on goes into the left-hand side panel of the office, not the main lobby."*

`st.header("📄 Knowledge Base")` just puts up a **section sign** on that side panel, like a wall sign reading "FILING DEPARTMENT."

`if not GEMINI_API_KEY:` — remember, this variable might be empty if the secret notebook (`.env`) was never set up properly (Section 3). This check is like a **big red warning sign** that immediately pops up: *"Hey, the secret access key is missing! Go add it to your notebook before anything else will work."* Much friendlier than letting the app confusingly fail somewhere deep inside, later, for no obvious reason.

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

**Simple version:**

- **`st.file_uploader(...)`** — this is the **drop-off tray** at the front desk where customers physically hand over their PDFs. `type=["pdf"]` means the tray only accepts PDFs — hand it a `.jpg` and it'll politely refuse. `accept_multiple_files=True` means you can drop off a whole stack at once, not just one at a time. Whatever's currently sitting in the tray becomes `uploaded_files`.

- **`st.text_input("Notify this email on upload", ...)`** — a little **sign-up sheet** where you write down an email address that should get a "your documents are ready!" notification once everything's processed. It comes pre-filled with your own email (as a sensible guess, since most people testing this will want to notify themselves), and hovering over the little "?" icon shows a helpful explanation of what this field actually does.

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

**Simple version — this is "what happens after you press the big red BUILD button":**

**`if st.button("Build / Rebuild Knowledge Base", disabled=not uploaded_files):`** — this is the button itself. It's **greyed out and unclickable** if nothing's been dropped in the tray yet (`disabled=not uploaded_files`) — no point letting someone press "build" on an empty pile. Everything below only happens on the exact moment this specific button gets clicked.

**`status_box = st.empty()`** — set up an **empty status board** on the wall that can be updated in place — like a "Now Serving..." digital sign at a deli counter, instead of printing a fresh new receipt for every single update.

**`def show_progress(msg): status_box.info(msg)`** — write a quick instruction: *"whenever someone hands you a status update, put it up on that board."* This little function gets handed all the way down into the batching/embedding process (Sections 7-9), so the customer watching the screen sees live updates like "Embedding chunks 51-100 of 320..." instead of a frozen, silent screen.

**Setting safe starting values before anything risky happens:**
```python
store, n_chunks, build_failed = None, 0, False
```
Like laying out "in case of emergency" defaults before starting a risky task — no matter what goes wrong below, these three values are guaranteed to exist in some sensible form.

**The actual attempt, with a safety net:**
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
**Attempt the whole filing/laminating process** (Section 9). If something goes seriously wrong that even the built-in retry logic (Section 8) couldn't fix, catch it here: flip a "something failed" flag, wipe the status board clean, and put up a **clear, specific error sign** — checking if it was the "too busy" problem specifically (giving better advice in that case) versus some other unexpected failure.

**`status_box.empty()`** (runs no matter what happened above) — clean the status board, since we're about to post a final result message instead.

**Deciding what final message to show:**
```python
if build_failed:
    pass  # error already shown above
elif store is None:
    st.error("No readable text found in the uploaded file(s).")
else:
    ...
```
- If it already failed with an error message shown above, there's nothing more to say — just move on.
- If nothing crashed, but the drawer still came back completely empty (`store is None`) — that means the PDFs had no readable text at all (maybe scanned images) — show a specific "couldn't read anything" message.
- Otherwise — genuine success! Move to the happy path below.

**On success:**
```python
st.session_state.store = store
st.session_state.messages = []
st.session_state.chat_history = []
st.success(f"Indexed {n_chunks} chunk(s) from {len(uploaded_files)} file(s).")
```
**Save the new drawer into permanent memory** (Section 21), **wipe the whiteboard clean** for a fresh conversation (since old chat history might reference documents that no longer exist in this new drawer), and **put up a green "success!" sign** with the final counts.

**Sending the "your documents are ready" letter:**
```python
if notify_email:
    ...
    with st.spinner(f"Sending notification email to {notify_email}..."):
        email_result = send_email(notify_email, subject, body)
    if email_result.startswith("Email successfully sent"):
        st.success(email_result)
    else:
        st.warning(email_result)
```
If someone filled in the notify-email sign-up sheet from Section 23, write up a friendly letter listing which files were processed and how many note cards were made, show a spinning "sending..." icon while it's in flight, and call the **plain mail clerk directly** — *not* through the AI at all, since this is just an automatic, guaranteed action the app always takes after a successful build, not something the librarian is deciding to do. Depending on what the clerk reports back, show either a green success sign or a yellow warning sign.

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

**Simple version:** `st.divider()` is just a **thin line drawn on the wall**, purely to visually separate one section of the sidebar from the next — no functional purpose, just tidiness.

`st.slider(...)` is a literal **drag-able volume knob** — except instead of controlling loudness, it controls **how picky the librarian is** about what counts as "relevant enough" (from Section 17). Slide it left (lower number), and the librarian becomes stricter, only trusting near-perfect matches. Slide it right (higher number), and the librarian becomes more lenient, willing to accept weaker matches. It starts at `0.8` by default, and whatever position it's in gets used the very next time someone asks a question.

---

## 26. Sidebar: New Chat button

```python
    st.divider()
    if st.button("🗑️ New Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
```

**Simple version:** This is the **"wipe the whiteboard clean and start a new conversation" button** — like erasing a whiteboard between two completely different meetings, but keeping all the filing cabinets (the actual indexed documents) exactly as they were.

`st.rerun()` at the end just means: *"Don't wait for the next thing to happen naturally — immediately redraw the screen right now,"* so the chat visibly empties out the instant you click, rather than looking stuck for a moment.

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

**Simple version:** This is the literal **"is the mail clerk on duty today?" switch**, and it's **off by default**. Nobody accidentally gets the mail-sending version of the librarian unless they specifically flip this switch on themselves — a sensible safety default, since sending real emails is a bigger deal than just answering a question.

---

## 28. Sidebar: chunk count caption

```python
    if st.session_state.store is not None:
        st.caption(f"Knowledge base: {st.session_state.store._collection.count()} chunks indexed")
```

**Simple version:** A tiny, quiet **inventory count label** at the bottom of the sidebar — like a small "Currently stocking: 320 items" sign — but only shown if there's actually a filing cabinet drawer to count in the first place. It's just a quick sanity-check number so you always know roughly how much is currently indexed.

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

**Simple version:** This is the **big lobby sign** at the front of the office (`st.title`) plus a small subtitle underneath explaining what this place actually does.

`if st.session_state.store is None:` — checks: *"has anyone actually filed any documents yet, ever?"* If not, instead of showing an empty, useless chat box, show a **friendly signpost** pointing the visitor toward the sidebar: *"Hey, go upload something first!"* Only once real documents exist does the actual chat interface (Sections 30-31) appear.

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

**Simple version:** Remember — **the entire script re-runs from scratch on every click.** That means there's no "permanent" chat window quietly sitting there waiting; the *entire visible conversation has to be completely redrawn from memory, every single time*, like an artist repainting the exact same picture from scratch on every heartbeat, using notes (`st.session_state.messages`) as the reference.

This loop walks through **every single message ever saved in that notebook** and redraws it:
- `with st.chat_message(msg["role"]):` — draw a **chat bubble**, automatically styled differently depending on whether it's from "user" or "assistant" (different avatar icon, alignment, etc.) — like choosing a different colored speech bubble for each speaker in a comic strip.
- `st.markdown(msg["content"])` — write the actual words inside that bubble, allowing for nicely formatted bold text, bullet points, etc. instead of raw plain text.
- **If this message has source citations attached:** draw a small **collapsible drawer labeled "Sources"** that's closed by default (`st.expander`) — clicking it reveals the little numbered source tags and their file/page labels (Section 18's citation chart), keeping the main conversation uncluttered unless someone actually wants to dig into the receipts.

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

**Simple version — this is the exact moment a customer asks a new question, from start to finish:**

**`user_input = st.chat_input(...)`** — this is the **text box glued to the bottom of the page** where the customer types their question. Nothing happens here on most screen redraws — it stays quiet (`None`) — *except* on the one specific moment right after someone actually presses Enter, at which point it hands over exactly what they typed.

**`if user_input:`** — only do anything below if a real question was actually just submitted.

**Show the customer's own question immediately:**
```python
st.session_state.messages.append({"role": "user", "content": user_input, "sources": None})
with st.chat_message("user"):
    st.markdown(user_input)
```
First, **write it into the permanent notebook** so it'll show up correctly on all future redraws (Section 30). But since this exact redraw already happened *before* this new question existed, we also have to **manually draw it right now**, immediately — otherwise the customer would type a question and see nothing happen for a moment, which feels broken.

**Now get the actual answer:**
```python
with st.chat_message("assistant"):
    with st.spinner("Thinking..."):
        llm = get_llm()
        answer, citation_map, standalone = answer_question(...)
```
Open up an assistant-style speech bubble, show a **"Thinking..." spinning icon** the whole time the manager (Section 20) is working through the entire process — condensing, searching, filtering, generating, and possibly mailing something.

**Show the results:**
```python
st.markdown(answer)
if citation_map:
    with st.expander("Sources"):
        for tag, label in citation_map.items():
            st.markdown(f"**{tag}** — {label}")
if standalone != user_input:
    st.caption(f"(interpreted as: \"{standalone}\")")
```
Print the actual answer, add the collapsible "Sources" drawer if there are any citations to show, and — this is a nice transparency touch — **if the question got secretly rewritten** behind the scenes (Section 13/16) into something different from what the customer actually typed, quietly show a small note underneath revealing *"(interpreted as: ...)"*, so nothing about that hidden rewriting step feels sneaky or confusing.

**Save everything into permanent memory for next time:**
```python
st.session_state.messages.append({"role": "assistant", "content": answer, "sources": citation_map})
st.session_state.chat_history.append(HumanMessage(content=user_input))
st.session_state.chat_history.append(AIMessage(content=answer))
```
Write the assistant's reply into the display notebook, and — importantly — write **the original, exactly-as-typed question** (not the secretly rewritten version) plus the final answer into the second, more minimal notebook used purely for future question-condensing. This keeps the *stored* conversation reading naturally, the way the customer actually experienced it, even though a slightly different, cleaned-up version was used briefly behind the scenes just for searching.

---

## 32. Putting it all together — the full request lifecycle

**Simple version — imagine watching this whole office in action, start to finish, one full day:**

1. **The office opens** (§21): staff are hired once and remembered (§4, §12) instead of re-hired constantly. Someone checks if there's a filing cabinet drawer already sitting around from yesterday (§6) — if so, it's quietly reopened, no re-filing needed.
2. **A customer drops off a stack of PDFs and presses "Build"** (§22-24): the papers get filed, photocopied, cut into note cards, laminated in careful batches (with polite retries if the laminator's busy — §7-9), and locked into a brand new labeled drawer. A "your documents are ready" letter optionally goes out (§10, §24).
3. **The customer asks a question** (§31): it's written on the board immediately.
4. **The manager takes over** (§20):
   - Cleans up the question if it's vague, using earlier chat (§13, §15, §16).
   - Digs through the filing cabinet, keeping only genuinely relevant note cards (§17).
   - Staples numbered source tabs onto whatever was found (§18).
   - **If the mail clerk switch is off** (§14, §20 first path): fills out the plain rulebook form, asks the librarian, gets a text answer back.
   - **If the mail clerk switch is on** (§19, §20 second path): the librarian decides whether to just answer, or to actually call the mail clerk (§11) to send a real letter (§10) — then reports back what happened, in plain words.
5. **The answer goes up on the board** (§31), along with its source tabs if any, and a quiet note if the question was secretly cleaned up first.
6. **Both notebooks get updated** (§31) so the *next* question has full context to work with — and the whole cycle repeats, with the entire office quietly "resetting and repainting itself from notes" (§4, §21) on every single interaction, in a way that *feels* completely continuous to the customer even though, technically, nothing is truly persistent except what's written in the notebook.

---

*Read this side-by-side with the actual `app.py` file — every numbered section here matches the same numbered section in the technical version, just explained the easy way.*
