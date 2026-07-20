from flask import Flask, render_template, jsonify, request
import threading
import os
from agent_runner import run_agent

app = Flask(__name__)

# Route to serve the UI
@app.route('/')
def index():
    return render_template('index.html')

# API to start the agent in a background thread
@app.route('/api/run', methods=['POST'])
def start_agent():
    # Run in a separate thread so it doesn't block the UI
    thread = threading.Thread(target=run_agent)
    thread.start()
    return jsonify({"status": "Agent started"})

# API to get logs
@app.route('/api/logs', methods=['GET'])
def get_logs():
    updates = ""
    final_post = ""
    
    if os.path.exists("updates.txt"):
        try:
            with open("updates.txt", "r", encoding='utf-8') as f:
                updates = f.read()
        except:
            pass
            
    if os.path.exists("final_post.txt"):
        try:
            with open("final_post.txt", "r", encoding='utf-8') as f:
                final_post = f.read()
        except:
            pass
            
    return jsonify({
        "updates": updates,
        "final_post": final_post
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
