from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embedding
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


embedding = download_hugging_face_embedding()

index_name = "medical-chatbot"
# Embed each chunk and upsert the embeddings into your Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    embedding = embedding,
    index_name = index_name
)




retriever = docsearch.as_retriever(search_type = "similarity",search_kwargs={"k":3})

chatmodel = ChatGroq(model = "llama-3.3-70b-versatile")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)


contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Given the chat history and the latest user question, "
         "rewrite the question into a standalone question. "
         "Do not answer the question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

history_aware_retriever = create_history_aware_retriever(
    chatmodel,
    retriever,
    contextualize_q_prompt
)

question_answer_chain = create_stuff_documents_chain(chatmodel, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)


store = {}

def get_session_history(session_id):

    if session_id not in store:

        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(

    rag_chain,

    get_session_history,

    input_messages_key="input",

    history_messages_key="chat_history",

    output_messages_key="answer"

)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = conversational_rag_chain.invoke({"input": msg},
        config={
            "configurable": {
            "session_id": "medical_user"
            }
        }

    )
    print("Response : ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0",port = 8080, debug=True)