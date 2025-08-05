from flask import Flask, request, jsonify
import google.generativeai as genai
from flask_cors import CORS 
import uuid
import os
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Remove in production or specify domains

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_api_key_here")
MODEL_NAME = "gemini-2.5-flash"
MAX_RESPONSE_TOKENS = 600
SYSTEM_PROMPT = "You are a mature friend to people in their teenage and respond to the queries that teenagers have in a friendly, easy to understand and concise way. Often using points and example limiting it to less than 350 tokens strictly. Use Proper formatting in the response so it's very clear to understand."

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(
    MODEL_NAME,
    system_instruction=SYSTEM_PROMPT,
    generation_config=genai.GenerationConfig(
        max_output_tokens=MAX_RESPONSE_TOKENS
    ),
    # Adjust safety settings to reduce blocking
    safety_settings={
        'HATE': 'BLOCK_NONE',
        'HARASSMENT': 'BLOCK_NONE',
        'SEXUAL': 'BLOCK_NONE',
        'DANGEROUS': 'BLOCK_NONE'
    }
)

# In-memory session storage
sessions = {}

class ChatSession:
    def __init__(self, session_id):
        self.id = session_id
        self.history = []
        self.created_at = datetime.now()
        self.last_used = datetime.now()
        self.chat = model.start_chat(history=[])
    
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

def extract_response_text(response):
    """Safely extract text from Gemini response with error handling"""
    try:
        # First try the standard method
        return response.text
    except ValueError:
        # If standard method fails, try manual extraction
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                    if text_parts:
                        return ''.join(text_parts)
        
        # Check for safety blocking
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            return "I'm sorry, I couldn't respond to that due to content safety restrictions."
        
        return "I'm sorry, I couldn't generate a response for that query."

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.json
    query = data.get('query')
    session_id = data.get('session_id')
    
    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400
    
    try:
        session = get_session(session_id)
        session.add_message("user", query)
        
        # Generate response
        response = session.chat.send_message(query)
        answer = extract_response_text(response)
        
        session.add_message("model", answer)
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
    now = datetime.now()
    expired_keys = [
        sid for sid, session in sessions.items()
        if (now - session.last_used).total_seconds() > max_age_seconds
    ]
    
    for key in expired_keys:
        sessions.pop(key, None)
    
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
        "system_prompt": SYSTEM_PROMPT,
        "max_response_tokens": MAX_RESPONSE_TOKENS,
        "capabilities": ["text", "multimodal"],
        "max_input_tokens": 1048576,
        "cost_effective": True
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)