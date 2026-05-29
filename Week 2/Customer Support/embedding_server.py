"""
embedding_server.py

Local HTTP server that serves HuggingFace embeddings to n8n.
n8n calls this instead of OpenAI for embeddings.

Usage:
  pip install sentence-transformers flask
  python embedding_server.py

Runs on: http://localhost:8000/embed
"""

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

print("Loading model (first run downloads ~90MB)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model ready! Server starting on http://localhost:8000")

@app.route('/embed', methods=['POST'])
def embed():
    data = request.json
    text = data.get('text', '')
    embedding = model.encode(text).tolist()
    return jsonify({ 'embedding': embedding })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({ 'status': 'ok' })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
