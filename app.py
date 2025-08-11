from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import os
from datetime import datetime
import nest_asyncio

# Apply the patch to allow nested asyncio event loops
nest_asyncio.apply()

# --- Langchain and Gemini Imports ---
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

app = Flask(__name__)
CORS(app)  # Configure for specific domains in production

# --- Configuration ---
os.environ["GOOGLE_API_KEY"] = "xxx"
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

DOCUMENT_PATH = "document.txt"

# --- Global RAG Agent Components ---
# These are initialized once when the application starts.
LLM = None
VECTORSTORE = None
RAG_CHAIN = None

def initialize_global_rag_agent():
    """
    Initializes the core RAG components that are shared across all sessions.
    This should be called only once at startup.
    """
    global LLM, VECTORSTORE, RAG_CHAIN
    
    print("Initializing global RAG components...")
    
    # 1. Initialize LLM and Embeddings
    LLM = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 2. Load, split, and create the vector store
    print("Setting up vector store...")
    loader = TextLoader(DOCUMENT_PATH)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    VECTORSTORE = FAISS.from_documents(documents=splits, embedding=embeddings)
    print(f"Vector store created with {len(splits)} document splits.")

    # 3. Create the conversational RAG chain
    print("Creating RAG chain...")
    # History-Aware Retriever
    contextualize_q_system_prompt = """Given a chat history and the latest user question, \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        LLM, VECTORSTORE.as_retriever(), contextualize_q_prompt
    )

    # Answering Chain
    qa_system_prompt = """You are a mature, non-judgmental friend/Psychologist for teenagers. \
    Provide helpful, honest advice on any topic while maintaining appropriate boundaries. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer from the context, say that you don't have information on that topic. \
    Strictly reply in 2 to 3 sentences or bullet points, no more than that.

    Context:
    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(LLM, qa_prompt)

    # Full RAG Chain
    RAG_CHAIN = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    print("Global RAG components initialized successfully.")


# --- Session Management Class ---
class ChatSession:
    """
    A lightweight class to hold the chat history for a single session.
    """
    def __init__(self, session_id):
        self.id = session_id
        self.chat_history = []

# In-memory session storage. For production, consider using Redis or a database.
sessions = {}

# --- Flask Endpoints ---
@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.json
    query = data.get('query')
    session_id = data.get('session_id', None)
    
    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400
    
    # Get or create a chat session
    if session_id and session_id in sessions:
        session = sessions[session_id]
    else:
        session_id = f"session_{uuid.uuid4()}"
        session = ChatSession(session_id)
        sessions[session_id] = session
        
    try:
        # Get the response from the RAG chain
        response_data = RAG_CHAIN.invoke({
            "input": query,
            "chat_history": session.chat_history
        })
        
        ai_response_text = response_data["answer"]

        # Update chat history for the session
        session.chat_history.append(HumanMessage(content=query))
        session.chat_history.append(AIMessage(content=ai_response_text))
        print("hi ----------")
        print({
            "response": ai_response_text,
            "session_id": session_id,
            "history_length": len(session.chat_history),
            "retrieved_context": [doc.page_content for doc in response_data.get("context", [])]
        })
        
        return jsonify({
            "response": ai_response_text,
            "session_id": session_id,
            "history_length": len(session.chat_history),
            "retrieved_context": [doc.page_content for doc in response_data.get("context", [])]
        })
    
    except Exception as e:
        print(f"An error occurred during chat: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Deletes a session to free up memory."""
    if session_id in sessions:
        sessions.pop(session_id, None)
        return jsonify({"status": f"Session {session_id} deleted successfully."})
    return jsonify({"error": "Session not found"}), 404

if __name__ == '__main__':
    # Ensure the knowledge base document exists
    if not os.path.exists(DOCUMENT_PATH):
        with open(DOCUMENT_PATH, "w") as f:
            f.write("This is a sample document for the RAG agent.\n")
            f.write("Friendship is about supporting each other.\n")
            
    # Initialize the global RAG agent before starting the app
    initialize_global_rag_agent()
    # app.run(host='0.0.0.0', port=5000, debug=True)
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
