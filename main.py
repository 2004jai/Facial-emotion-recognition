import os
import subprocess
import threading
import time
import webbrowser

def start_flask_server():
    """Start the Flask API server in a separate process."""
    print("Starting Flask API server...")
    subprocess.run(["python", "flask.py"])

def start_streamlit_app():
    """Start the Streamlit app in a separate process."""
    print("Starting Streamlit app...")
    subprocess.run(["streamlit", "run", "web.py"])

def main():
    """Main function to start both servers."""
    print("Starting Emotion Detection Web Application")
    
    # Check if required files exist
    required_files = ["face_landmarks.dat", "emotion.h5"]
    missing_files = [file for file in required_files if not os.path.exists(file)]
    
    if missing_files:
        print("Error: The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease download these files before running the application.")
        return
    
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=start_flask_server)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Wait for Flask server to start
    print("Waiting for Flask server to start...")
    time.sleep(5)
    
    # Start Streamlit app
    start_streamlit_app()

if __name__ == "__main__":
    main()