
# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import os
import uuid

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash
)

from dotenv import load_dotenv
from werkzeug.utils import secure_filename

from pypdf import PdfReader

from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings
)


from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


# ==========================================
# 2. ENVIRONMENT CONFIGURATION
# ==========================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing.")

CHAT_MODEL = os.getenv(
    "OPENAI_CHAT_MODEL",
    "gpt-4.1-mini"
)

EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small"
)


# ==========================================
# 3. FLASK CONFIGURATION
# ==========================================

app = Flask(__name__)

app.secret_key = os.getenv("SECRET_KEY")

if not app.secret_key:
    raise ValueError("SECRET_KEY is missing.")

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

DATA_DIR = "__data__"

os.makedirs(DATA_DIR, exist_ok=True)


# ==========================================
# 4. INITIALIZE OPENAI MODELS
# ==========================================

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=OPENAI_API_KEY
)

llm = ChatOpenAI(
    model=CHAT_MODEL,
    temperature=0.3,
    api_key=OPENAI_API_KEY
)



# ==========================================
# 5. STORE USER VECTOR DATABASES
# ==========================================

vectorstores = {}


def get_user_id():

    if "user_id" not in session:

        session["user_id"] = str(uuid.uuid4())

    return session["user_id"]


# ==========================================
# 6. PDF TEXT EXTRACTION
# ==========================================

def get_pdf_text(pdf_docs):

    extracted_text = ""

    for pdf in pdf_docs:

        if not pdf or not pdf.filename:
            continue

        filename = secure_filename(pdf.filename)

        if not filename.lower().endswith(".pdf"):
            continue

        pdf_id = str(uuid.uuid4())

        file_path = os.path.join(
            DATA_DIR,
            f"{pdf_id}_{filename}"
        )

        pdf.stream.seek(0)

        reader = PdfReader(pdf.stream)

        pdf_text = ""

        for page in reader.pages:

            page_text = page.extract_text() or ""

            pdf_text += page_text + "\n"

        if not pdf_text.strip():
            continue

        with open(file_path, "w", encoding="utf-8") as file:

            file.write(pdf_text)

        extracted_text += pdf_text + "\n"

    return extracted_text


# ==========================================
# 7. TEXT CHUNKING
# ==========================================

def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=200,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],

        length_function=len

    )

    chunks = text_splitter.split_text(text)

    return chunks


# ==========================================
# 8. CREATE FAISS VECTOR DATABASE
# ==========================================

def get_vectorstore(text_chunks):

    vectorstore = FAISS.from_texts(

        texts=text_chunks,

        embedding=embeddings

    )

    return vectorstore


# ==========================================
# 9. RETRIEVE RELEVANT DOCUMENTS
# ==========================================

def retrieve_documents(vectorstore, question):

    documents = vectorstore.similarity_search(

        question,

        k=4

    )

    context = "\n\n".join(

        [document.page_content for document in documents]

    )

    return context


# ==========================================
# 10. CONVERSATIONAL RAG
# ==========================================

def get_conversation_response(
    question,
    vectorstore,
    chat_history
):

    # Convert follow-up questions into
    # standalone questions for retrieval.

    if chat_history:

        history_text = "\n".join(

            [
                f"{message['role']}: {message['content']}"
                for message in chat_history[-6:]
            ]

        )

        rewrite_prompt = [

            SystemMessage(
                content=(
                    "Rewrite the user's latest question as a "
                    "standalone question using the conversation "
                    "history. Do not answer the question."
                )
            ),

            HumanMessage(
                content=(
                    f"Conversation history:\n{history_text}\n\n"
                    f"Latest question: {question}"
                )
            )

        ]

        standalone_question = llm.invoke(
            rewrite_prompt
        ).content

    else:

        standalone_question = question


    # Retrieve relevant document chunks.

    context = retrieve_documents(

        vectorstore,

        standalone_question

    )


    # Construct the final prompt.

    system_prompt = """

    You are Chat Scholar, an academic AI assistant.

    Answer the student's questions using the
    provided document context.

    Instructions:

    1. Give accurate and understandable answers.
    2. Explain complex concepts in simple language.
    3. Use examples wherever necessary.
    4. Do not invent information.
    5. If the answer is not available in the
       provided documents, clearly mention that.
    6. Use conversation history when appropriate.

    DOCUMENT CONTEXT:

    {context}

    """

    messages = [

        SystemMessage(
            content=system_prompt.format(
                context=context
            )
        )

    ]


    # Add previous conversation messages.

    for message in chat_history[-6:]:

        if message["role"] == "user":

            messages.append(

                HumanMessage(
                    content=message["content"]
                )

            )

        elif message["role"] == "assistant":

            messages.append(

                AIMessage(
                    content=message["content"]
                )

            )


    # Add current question.

    messages.append(

        HumanMessage(
            content=question
        )

    )


    # Generate answer.

    response = llm.invoke(messages)

    return response.content


# ==========================================
# 11. ESSAY GRADING
# ==========================================

def grade_essay(essay, rubric):

    system_prompt = f"""

    You are an experienced academic essay evaluator.

    Evaluate the student's essay using the
    following grading rubric.

    RUBRIC:

    {rubric}

    Evaluation instructions:

    1. Evaluate the essay according to the rubric.
    2. Identify strengths.
    3. Identify weaknesses.
    4. Evaluate grammar and vocabulary.
    5. Check coherence and structure.
    6. Provide constructive suggestions.
    7. Give a score based on the rubric.
    8. Explain why the score was awarded.

    Do not invent grading criteria that
    contradict the provided rubric.

    Return the evaluation in a structured format.

    """

    messages = [

        SystemMessage(
            content=system_prompt
        ),

        HumanMessage(
            content=f"Evaluate this essay:\n\n{essay}"
        )

    ]

    response = llm.invoke(messages)

    return response.content


# ==========================================
# 12. EXTRACT ESSAY FROM PDF
# ==========================================

def extract_text_from_pdf(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        text += (page.extract_text() or "") + "\n"

    return text


# ==========================================
# 13. HOME ROUTE
# ==========================================

@app.route("/")
def home():

    return render_template("new_home.html")


# ==========================================
# 14. PROCESS UPLOADED DOCUMENTS
# ==========================================

@app.route("/process", methods=["POST"])
def process_documents():

    pdf_docs = request.files.getlist("pdf_docs")

    raw_text = get_pdf_text(pdf_docs)

    if not raw_text.strip():

        flash("No readable PDF content found.")

        return redirect(url_for("home"))


    # Step 1: Split documents.

    text_chunks = get_text_chunks(raw_text)


    # Step 2: Generate embeddings and FAISS index.

    vectorstore = get_vectorstore(text_chunks)


    # Step 3: Associate vectorstore with user.

    user_id = get_user_id()

    vectorstores[user_id] = vectorstore


    # Step 4: Reset conversation history.

    session["chat_history"] = []


    return redirect(url_for("chat"))


# ==========================================
# 15. CHAT ROUTE
# ==========================================

@app.route("/chat", methods=["GET", "POST"])
def chat():

    user_id = get_user_id()

    chat_history = session.get(
        "chat_history",
        []
    )


    if user_id not in vectorstores:

        flash("Please upload PDF documents first.")

        return redirect(url_for("home"))


    if request.method == "POST":

        user_question = request.form.get(
            "user_question",
            ""
        ).strip()


        if user_question:

            vectorstore = vectorstores[user_id]


            response = get_conversation_response(

                user_question,

                vectorstore,

                chat_history

            )


            chat_history.append({

                "role": "user",

                "content": user_question

            })


            chat_history.append({

                "role": "assistant",

                "content": response

            })


            session["chat_history"] = chat_history


    return render_template(

        "new_chat.html",

        chat_history=chat_history

    )


# ==========================================
# 16. PDF CHAT PAGE
# ==========================================

@app.route("/pdf_chat")
def pdf_chat():

    return render_template(
        "new_pdf_chat.html"
    )


# ==========================================
# 17. ESSAY GRADING ROUTE
# ==========================================

@app.route("/essay_grading", methods=["GET", "POST"])
def essay_grading():

    result = None

    input_text = ""

    if request.method == "POST":

        rubric = request.form.get(
            "essay_rubric",
            ""
        ).strip()


        if rubric:

            session["rubric_text"] = rubric


        rubric = session.get(
            "rubric_text",
            ""
        )


        pdf_file = request.files.get("file")


        if pdf_file and pdf_file.filename:

            input_text = extract_text_from_pdf(
                pdf_file
            )

        else:

            input_text = request.form.get(
                "essay_text",
                ""
            ).strip()


        if not rubric:

            flash("Please provide an essay grading rubric.")

        elif not input_text.strip():

            flash("Please provide an essay.")

        else:

            result = grade_essay(

                input_text,

                rubric

            )


    return render_template(

        "new_essay_grading.html",

        result=result,

        input_text=input_text

    )


# ==========================================
# 18. ESSAY RUBRIC PAGE
# ==========================================

@app.route("/essay_rubric")
def essay_rubric():

    return render_template(
        "new_essay_rubric.html"
    )


# ==========================================
# 19. RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )