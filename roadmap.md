# RAG Document Assistant -- Learning Roadmap

This roadmap explains the project in the order it executes, rather than
by function.

------------------------------------------------------------------------

# Stage 0 -- Setup

## Goal

Initialize everything the application needs before doing any work.

### Topics

-   Environment variables (`.env`)
-   Gemini API
-   Embedding model
-   LLM
-   Streamlit
-   Chroma

### Functions

-   `load_dotenv()`
-   `get_embeddings_model()`
-   `get_llm()`
-   `get_llm_with_tools()`

### Flow

``` text
API Keys
    │
    ▼
Embedding Model

LLM

LLM + Email Tool
```

------------------------------------------------------------------------

# Stage 1 -- Build the Knowledge Base

Runs only when the user uploads PDFs.

## Step 1 -- Upload PDFs

**Topic** - Streamlit File Uploader

**Input**

    uploaded_files

------------------------------------------------------------------------

## Step 2 -- Save PDFs

**Function** - `build_store_from_uploads()`

**Purpose** Save uploaded PDFs into the `data/` folder.

------------------------------------------------------------------------

## Step 3 -- Load PDFs

**Topic** - `PyPDFLoader`

**Function** - `PyPDFLoader(path).load()`

### Flow

``` text
PDF
 │
 ▼
Documents
```

------------------------------------------------------------------------

## Step 4 -- Chunking

**Topic** - `RecursiveCharacterTextSplitter`

**Function** - `split_documents()`

### Flow

``` text
Document
 │
 ▼
Chunk 1
Chunk 2
Chunk 3
```

------------------------------------------------------------------------

## Step 5 -- Embeddings

**Topic** - Google Embedding Model

**Function** - `get_embeddings_model()`

### Flow

``` text
Text
 │
 ▼
Vector
```

------------------------------------------------------------------------

## Step 6 -- Store in Chroma

**Topic** - Vector Database

**Functions** - `build_vector_store_batched()` -
`_add_batch_with_retry()`

### Flow

``` text
Chunks
 │
 ▼
Embeddings
 │
 ▼
Chroma Vector Database
```

Knowledge base is now ready.

------------------------------------------------------------------------

# Stage 2 -- User Asks a Question

## Step 1 -- Receive User Question

Input:

``` text
How does LangChain work?
```

------------------------------------------------------------------------

## Step 2 -- Conversational Memory

**Topics** - Conversation History - Question Condensing

**Functions** - `format_history()` - `condense_question()`

### Flow

``` text
Previous Chat
      +
Current Question
      │
      ▼
Standalone Question
```

------------------------------------------------------------------------

## Step 3 -- Retrieval

**Topic** - Similarity Search

**Function** - `retrieve_filtered()`

### Flow

``` text
Standalone Question
      │
      ▼
Embedding
      │
      ▼
Similarity Search
      │
      ▼
Top Matching Chunks
```

------------------------------------------------------------------------

## Step 4 -- Filtering

Only keep chunks whose similarity score is below the chosen threshold.

------------------------------------------------------------------------

## Step 5 -- Build Context

**Function** - `format_docs_with_citations()`

Outputs:

-   Context
-   Citation Map

Example:

``` text
[Source 1]
...

[Source 2]
...
```

------------------------------------------------------------------------

# Stage 3 -- Generate the Answer

**Function** - `generate_grounded_answer()`

### Flow

``` text
Context
    +
Question
    │
    ▼
Prompt
    │
    ▼
Gemini
    │
    ▼
Answer
```

If no relevant context exists:

``` text
Fallback Prompt
        │
        ▼
General Knowledge Answer
```

------------------------------------------------------------------------

# Stage 4 -- Agent Decision

Inside:

-   `answer_question()`

Decision Flow

``` text
Agent Enabled?
      │
      ▼
Does user want email?
      │
  Yes │ No
      ▼
```

If **No**

``` text
Normal RAG
```

If **Yes**

``` text
LLM with Tools
      │
      ▼
Tool Calling
      │
      ▼
SMTP Email
      │
      ▼
Final Response
```

------------------------------------------------------------------------

# Stage 5 -- Streamlit UI

Responsible only for displaying information.

Main components:

-   `st.chat_input()`
-   `st.chat_message()`
-   `st.session_state`

Flow:

``` text
User
 │
 ▼
Streamlit
 │
 ▼
answer_question()
 │
 ▼
Display Answer
```

------------------------------------------------------------------------

# Complete Pipeline

``` text
                 BUILD PHASE

Upload PDF
      │
      ▼
Save PDF
      │
      ▼
PyPDFLoader
      │
      ▼
Chunking
      │
      ▼
Embedding Model
      │
      ▼
Embeddings
      │
      ▼
Chroma Database

────────────────────────────────────

                 QUERY PHASE

User Question
      │
      ▼
Conversation Memory
(format_history +
condense_question)
      │
      ▼
Standalone Question
      │
      ▼
Retriever
      │
      ▼
Similarity Search
      │
      ▼
Relevant Chunks
      │
      ▼
Context Builder
      │
      ▼
Prompt
      │
      ▼
Gemini
      │
      ▼
Answer

────────────────────────────────────

(Optional)

Enable Agent
      │
      ▼
Need Email?
      │
      ▼
Tool Calling
      │
      ▼
SMTP
      │
      ▼
Final Answer
```

------------------------------------------------------------------------

# Topics to Master

1.  Python basics
2.  Streamlit
3.  PyPDFLoader
4.  Chunking
5.  Embeddings
6.  Chroma Vector Database
7.  Retrieval
8.  Prompt Engineering
9.  LangChain Chains
10. Conversational Memory
11. RAG Answer Generation
12. Tool Calling
13. SMTP Email
14. Session State

Mastering these topics will let you understand nearly every part of this
project.
