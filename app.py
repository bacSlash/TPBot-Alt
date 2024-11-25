from flask import Flask, render_template, request, jsonify, session
import openai
import os
import faiss
import numpy as np
from dotenv import load_dotenv
import pickle
import uuid
from flask_socketio import SocketIO

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
app.config['SESSION_TYPE'] = 'filesystem'

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow CORS for Render deployment

# Path to store cached embeddings
EMBEDDINGS_CACHE_FILE = 'embeddings_cache.pkl'

# Store user queries and their embeddings in a dictionary for caching
embedding_cache = {}
user_queries = {}

def load_embeddings_from_cache():
    if os.path.exists(EMBEDDINGS_CACHE_FILE):
        with open(EMBEDDINGS_CACHE_FILE, 'rb') as f:
            # Load as a tuple with (faiss_index, sections)
            faiss_index, sections = pickle.load(f)
            return faiss_index, sections
    return None, None

def get_embedding(text, model="text-embedding-ada-002"):
    if text in embedding_cache:
        return embedding_cache[text]

    # Update this line according to the new API usage
    response = openai.Embedding.create(input=[text], model=model)  # Updated API call with list format

    embedding = response['data'][0]['embedding']
    embedding_cache[text] = embedding
    return embedding

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_query = data.get('query')

    if not user_query:
        return jsonify({'response': "Invalid query: query cannot be empty"})

    user_id = session.get('user_id')
    if not user_id:
        user_id = str(uuid.uuid4())
        session['user_id'] = user_id

    if user_id not in user_queries:
        user_queries[user_id] = []
    user_queries[user_id].append(user_query)

    if user_query.lower() == "what was my previous question?":
        previous_question = user_queries[user_id][-2] if len(user_queries[user_id]) > 1 else None
        if previous_question:
            return jsonify({'response': f"Your previous question was: {previous_question}"})
        else:
            return jsonify({'response': "I don't have any previous questions."})

    faiss_index, sections = load_embeddings_from_cache()

    if faiss_index is None or sections is None:
        return jsonify({'response': "Embeddings not found. Please generate them first."})

    try:
        query_embedding = get_embedding(user_query)
        distances, indices = faiss_index.search(np.array([query_embedding], dtype=np.float32), k=3)
        relevant_sections = [sections[i] for i in indices[0] if i != -1]

        if relevant_sections:
            relevant_content = ' '.join(relevant_sections)
            gpt_response = generate_gpt_response(relevant_content, user_query)
            return jsonify({'response': gpt_response})
        else:
            return jsonify({'response': "No relevant content found."})

    except Exception as e:
        return jsonify({'response': f"Error processing query: {e}"})

def generate_gpt_response(context, query):
    prompt = f"""  
You are TPBot, an AI assistant excusively for Teleperformance, a global leader in digital business services.
Your primary function is to assist with inquiries about Teleperformance's products, services, and company information in multiple languages.
Use the following content to answer the user's questions: {context}
IMPORTANT INSTRUCTIONS:
1. Be conversational, polite, and adaptive. Respond appropriately to greetings, small talk and Teleperformance related queries.
2. For greetings or small talk, engage briefly and natually, then guide the coversation towards Teleperformance topics.
3. Keep responses concise, professional, and short, typically within 2-3 sentences unless more detail is necessary.
4. Use only the provided context for Teleperformance related information. Don't invent or assume details.
5. If a question isn't about Teleperformance, politely redirect: 'I apologize, but I can only provide information about Teleperformance, its products, and services. Is there anythiung else you'd like to know?'
6. For unclear questions, ask for clarification: 'To ensure I provide the most accurate information about Teleperformance, could you please rephrase your question?'
7. Adjust your language style to match the user's - formal or casual - but always maintain professionalism.
8. Always respond in the sam elanguage as the user's input.
9. If the context doesn't provide enough information for a comprehensice answer, be honest about the limitation and offer to assit with related topics you can confidently address.
10. Remember previous interactions within the conversation and maintain contenxt continuity. 
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use 'gpt-4' or 'gpt-3.5-turbo' depending on your access
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ],
        max_tokens=100  # Limit the response length
    )
    
    return response['choices'][0]['message']['content']

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
