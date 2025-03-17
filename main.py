import os
import threading
import streamlit as st
from flask import Flask, request, jsonify

# Initialize Flask
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Flask API is running!"})

# Function to run Streamlit
def run_streamlit():
    os.system("streamlit run main.py --server.port 8501 --server.address 0.0.0.0")

# Run Streamlit in a separate thread
threading.Thread(target=run_streamlit, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
