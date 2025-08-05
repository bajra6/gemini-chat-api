from flask import Flask, request, jsonify
import google.generativeai as genai
from flask_cors import CORS 
import uuid
import os
import time
from datetime import datetime

app = Flask(__name__)


CORS(app)# Remove "*" in production environments

# Configuration (update with your API key)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_api_key_here")
MODEL_NAME = "gemini-2.5-flash"  # Latest stable version :cite[2]:cite[3]
MAX_RESPONSE_TOKENS = 250
SYSTEM_PROMPT = "You are a mature friend to people in their teenage and respond to the queries that teenagers have in a friendly, easy to understand and concise way. Often using points and example limiting it to less than 150 words. Use Proper formatting in the response so it's very clear to understand."

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY, )
model = genai.GenerativeModel(
    MODEL_NAME,
    system_instruction=SYSTEM_PROMPT,  # Correct parameter placement
    generation_config=genai.GenerationConfig(
        max_output_tokens=MAX_RESPONSE_TOKENS
    )
)
# In-memory session storage (replace with Redis in production)
sessions = {}

class ChatSession:
    def __init__(self, session_id):
        self.id = session_id
        self.history = []
        self.created_at = datetime.now()
        self.last_used = datetime.now()
        self.chat = model.start_chat(
            history=[]
        )
    
    def add_message(self, role, parts):
        self.history.append({"role": role, "content": parts, "timestamp": time.time()})
        self.last_used = datetime.now()
    
    def get_gemini_history(self):
        return [{"role": msg["role"], "parts": [msg["content"]]} for msg in self.history]

def get_session(session_id=None):
    if session_id and session_id in sessions:
        session = sessions[session_id]
        session.last_used = datetime.now()
        return session
    
    new_id = session_id or f"session_{uuid.uuid4()}"
    session = ChatSession(new_id)
    sessions[new_id] = session
    return session

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.json
    query = data.get('query')
    session_id = data.get('session_id')
    
    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400
    
    try:
        # Get or create session
        session = get_session(session_id)
        
        # Add user message to history
        session.add_message("user", query)
        
        # Generate response
        response = session.chat.send_message(query)
        answer = response.text
        
        # Add model response to history
        session.add_message("model", answer)
        
        # Manage session storage (basic cleanup)
        cleanup_old_sessions()
        
        return jsonify({
            "response": answer,
            "session_id": session.id,
            "model": MODEL_NAME,
            "history_length": len(session.history)
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "session_id": session_id
        }), 500

def cleanup_old_sessions(max_age_seconds=3600, max_sessions=50):
    """Basic session cleanup to prevent memory bloat"""
    now = datetime.now()
    expired_keys = [
        sid for sid, session in sessions.items()
        if (now - session.last_used).total_seconds() > max_age_seconds
    ]
    
    for key in expired_keys:
        sessions.pop(key, None)
    
    # Enforce maximum session count
    if len(sessions) > max_sessions:
        oldest = sorted(sessions.items(), key=lambda x: x[1].last_used)[0][0]
        sessions.pop(oldest, None)

@app.route('/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    sessions.pop(session_id, None)
    return jsonify({"status": f"Session {session_id} deleted"})

@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        "model": MODEL_NAME,
        "capabilities": ["text", "multimodal"],
        "max_input_tokens": 1048576,  # 1 million tokens :cite[2]:cite[3]
        "max_output_tokens": 65536,
        "cost_effective": True  # Gemini 2.5 Flash is budget-friendly :cite[5]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)