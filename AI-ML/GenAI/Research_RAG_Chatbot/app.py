
# pip install -U langchain langchain-openai langchain-community langchain-text-splitters faiss-cpu pypdf python-dotenv

import os
import getpass
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# --- API key: load from .env, fall back to a prompt ---
load_dotenv()  # reads .env in the current folder into os.environ
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key: ")

# --- LLM ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print(llm.invoke("Explain EDA in just 2 lines").content)


# --- Load and split the PDF (from GitHub raw URL) ---
PDF_URL = "https://raw.githubusercontent.com/zep-analytics/GenAI-Project-ResearchRagCHAT/main/RAGPaper.pdf"

documents = PyPDFLoader(PDF_URL).load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# --- Embeddings + vector store (FAISS), cached on disk ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
index_dir = Path(__file__).parent / "faiss_index"

if index_dir.exists():
    db = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
else:
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(index_dir))

retriever = db.as_retriever(search_kwargs={"k": 4})

# --- Step 1: rewrite follow-ups into standalone questions ---
condense_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and a follow-up question, rewrite the "
               "follow-up as a standalone question. Return only the question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])
condense_chain = condense_prompt | llm | StrOutputParser()

# --- Step 2: answer using retrieved context ---
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using only the context below. "
               "If the answer isn't in the context, say you don't know.\n\n"
               "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])
answer_chain = answer_prompt | llm | StrOutputParser()

def ask(question, chat_history):
    standalone = (
        condense_chain.invoke({"question": question, "chat_history": chat_history})
        if chat_history else question
    )
    docs = retriever.invoke(standalone)
    context = "\n\n".join(d.page_content for d in docs)
    answer = answer_chain.invoke({
        "context": context,
        "question": question,
        "chat_history": chat_history,
    })
    return answer, docs

# --- Use it ---
chat_history = []

q1 = "What are RAGs and what are their use cases? Explain in detail."
a1, sources = ask(q1, chat_history)
print(a1)
chat_history += [HumanMessage(content=q1), AIMessage(content=a1)]

q2 = "Which of those use cases is most useful for question answering?"
a2, sources = ask(q2, chat_history)
print(a2)

for d in sources:
    print(d.metadata.get("page"), d.page_content[:100])