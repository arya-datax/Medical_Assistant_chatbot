from flask import Flask, render_template, request
import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.helper import download_embedding_model
from src.promt import prompt


# -------------------------
# Environment & App Setup
# -------------------------

def load_environment():
    load_dotenv()

    pinecone_key = os.getenv("PINECONE_API_KEY")
    genai_key = os.getenv("GENAI_API_KEY")

    if not pinecone_key:
        raise EnvironmentError("PINECONE_API_KEY is missing in environment variables")

    if not genai_key:
        raise EnvironmentError("GENAI_API_KEY is missing in environment variables")

    os.environ["PINECONE_API_KEY"] = pinecone_key
    os.environ["GENAI_API_KEY"] = genai_key



def create_app():
    app = Flask(__name__)
    return app


# -------------------------
# Vector DB & Embeddings
# -------------------------

def load_embeddings():
    return download_embedding_model()


def load_vectorstore(embedding):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = "medical-bot-minillm"

    return PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
        namespace=None
    )


def load_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )


# -------------------------
# LLM & RAG Chain
# -------------------------

def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.environ["GENAI_API_KEY"]
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever, llm):
    final_prompt = ChatPromptTemplate.from_template(prompt)

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | final_prompt
        | llm
    )


# -------------------------
# App Initialization
# -------------------------

load_environment()
app = create_app()

embedding = load_embeddings()
vectorstore = load_vectorstore(embedding)
retriever = load_retriever(vectorstore)
llm = load_llm()
rag_chain = build_rag_chain(retriever, llm)


# -------------------------
# Flask Routes
# -------------------------

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    response = rag_chain.invoke(msg)
    return response.content



# -------------------------
# Run Server
# -------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
