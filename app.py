from flask import Flask, render_template, jsonify, request
from src.helper import download_embedding_model
from pinecone import Pinecone  
from langchain_pinecone import PineconeVectorStore
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
from src.promt import *

app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GENAI_API_KEY=os.environ.get('GENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GENAI_API_KEY"] = GENAI_API_KEY

embedding = download_embedding_model()


# loading the existing index
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("medical-bot-minillm")

index_name = "medical-bot-minillm"
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding,
    namespace=None
)

retrieval = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":3})


# loading llm model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=GENAI_API_KEY)

# prompt template
final_prompt = ChatPromptTemplate.from_template(prompt)

# format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# rag chain
rag_chain = (
    {"context" : retrieval | format_docs,
    "question" : RunnablePassthrough()
    } | final_prompt | llm
)

res = rag_chain.invoke("What is Acne and how i can reduce it from my face?")


@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)