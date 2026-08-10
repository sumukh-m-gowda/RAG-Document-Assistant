from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = PyPDFLoader("file.pdf").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", google_api_key=YOUR_KEY)

store = Chroma(persist_directory="chroma_db", embedding_function=embeddings, collection_name="my_docs")
store.add_documents(documents)

results = store.similarity_search_with_score("wht is the name of the candidate")
for doc,score in results:
    print(score,doc.page_content[:100])

def retrieve_filtered(store , query ,k=5 , threshold=DEFAULT_THRESHOLD):
    result = store.similarity_search_with_score(query , k=k)
    return [(doc, score) for doc,score in results if score <= threshold]

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

ansswer_prompt = ChatPromptTemplate.from_template("""
answer using only the content given

content : {context}
question : {question}

answer:
""")
chain = ansswer_prompt | llm | StrOutputParser()
print(chain.invoke{"context" : "...chunk text here...", "question" : "what does it say about candidate profile"})

CONDENSE_PROMPT = ChatPromptTemplate.from_template("""
Rewrite the follow-up question to be standalone, using the history.

History:
{chat_history}

Follow-up: {question}

Standalone question:
""")

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

standalone = condense_question(question, chat_history, llm)
filtered = retrieve_filtered(store, standalone, threshold=threshold)
context, citation_map = format_docs_with_citations(filtered)

from langchain_core.tools import tool

@tool
def email_this(recipient_email:str ,subject:str , message:str) ->str :
    """"Send a email to the given reciepient with the given subject and message.
    use only when a user ask u email/mail/send something to address."""
    return send_email(recipient_email, subject, message)

EMAIL_KEYWORDS = ("email", "mail", "send")

def check_wants_email(question, agent_mode):
    return agent_mode and any(w in question.lower() for w in EMAIL_KEYWORDS)

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(recipient_email: str, subject: str, body: str) -> str:
    """Send an email via Gmail SMTP. Returns a success or error message (never raises)."""
    sender_email = os.environ.get('SENDER_EMAIL')
    app_password = os.environ.get('SENDER_APP_PASSWORD')

    try: 
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP_SSL('smtp.gmail.com',465) as server:
            server.login(sender_email,app_password)
            server.sendmail(sender_email,recipient_email,msg.as_string())

        return f'Email successfully sent to {recipient_email} with subject: {subject}'

    except smtplib.SMTPAuthenticationError:
        return 'Authentication failed. Make sure you are using a Gmail App Password, not your regular password.'
    except smtplib.SMTPRecipientsRefused:
        return f'Recipient {recipient_email} was refused. Check if the email address is correct.'
    except Exception as e:
        return f'Failed to send email: {str(e)}'

    