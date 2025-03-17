import streamlit as st
import cv2
import numpy as np
import requests
import json
import base64
from io import BytesIO
from PIL import Image
import time
import os

# Configure the API endpoint - change this to your deployed API URL
API_URL = os.environ.get("API_URL", "http://localhost:5000")


def get_api_status():
    """Check if the API is available"""
    try:
        response = requests.get(f"{API_URL}/api/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {
            "status": "error",
            "message": f"API returned status code {response.status_code}",
        }
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Cannot connect to API: {str(e)}"}


def encode_image_to_base64(image):
    """Convert an image (PIL Image or numpy array) to base64 string"""
    if isinstance(image, np.ndarray):
        # Convert numpy array to PIL Image
        image = Image.fromarray(image)

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"


def decode_base64_to_image(base64_string):
    """Convert a base64 string to a PIL Image"""
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]

    img_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(img_data))


def compress_image_for_api(image, quality=85, max_size=(800, 800)):
    """Compress an image to reduce API transmission size"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Resize if larger than max_size
    width, height = image.size
    if width > max_size[0] or height > max_size[1]:
        image.thumbnail(max_size, Image.LANCZOS)

    # Compress
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    compressed_image = Image.open(buffer)

    return compressed_image


def process_image_with_api(image):
    """Send an image to the API for processing and return the results"""
    # Compress image to reduce transfer size
    compressed_image = compress_image_for_api(image)

    # Convert image to base64
    base64_image = encode_image_to_base64(compressed_image)

    # Prepare API request
    payload = {"image": base64_image}

    try:
        with st.spinner("Processing image..."):
            # Send request to API
            response = requests.post(
                f"{API_URL}/api/detect",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,  # Increased timeout for processing
            )

            # Check if request was successful
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None


def process_video_frame_with_api(frame):
    """Send a video frame to the API for processing and return the results"""
    # Compress frame to reduce transfer size
    compressed_frame = compress_image_for_api(frame, quality=70, max_size=(640, 480))

    # Convert frame to base64
    base64_frame = encode_image_to_base64(compressed_frame)

    # Prepare API request
    payload = {"frame": base64_frame}

    try:
        # Send request to API
        response = requests.post(
            f"{API_URL}/api/video",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        # Check if request was successful
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None


def main():
    st.set_page_config(
        page_title="Emotion Detection App", page_icon="😀", layout="wide"
    )

    st.title("Real-Time Emotion Detection")

    # Check API status
    api_status = get_api_status()
    status_color = "green" if api_status.get("status") == "ok" else "red"
    st.sidebar.markdown(
        f"<h3>API Status: <span style='color:{status_color};'>{api_status.get('status', 'unknown')}</span></h3>",
        unsafe_allow_html=True,
    )

    if api_status.get("status") != "ok":
        st.error(f"API Error: {api_status.get('message', 'Unknown error')}")
        st.info("Please make sure the API server is running and accessible.")
        return

    # Add information about the app
    with st.expander("About this app"):
        st.markdown(
            """
        This application uses machine learning to detect emotions from faces in images and video streams.
        
        **Supported emotions:**
        - Happy 😀
        - Sad 😢
        - Angry 😠
        - Surprised 😲
        - Neutral 😐
        - Fear 😨
        - Disgust 🤢
        
        The app works by sending images to a backend API which processes them and returns the detected emotions.
        """
        )

    # Create tabs for different modes
    tab1, tab2, tab3 = st.tabs(
        ["Upload Image", "Webcam Capture", "Real-time Detection"]
    )

    # Tab 1: Upload Image
    with tab1:
        st.header("Upload an Image")
        uploaded_file = st.file_uploader(
            "Choose an image file", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Process button
            if st.button("Detect Emotions", key="process_image_btn"):
                # Process the image with the API
                results = process_image_with_api(image)

                if results:
                    # Display the processed image with annotations
                    if "processed_image" in results:
                        processed_img = decode_base64_to_image(
                            results["processed_image"]
                        )
                        st.image(
                            processed_img,
                            caption="Processed Image",
                            use_column_width=True,
                        )

                    # Display detection results
                    if "detections" in results and results["detections"]:
                        st.subheader("Detected Emotions")

                        # Create columns for each detection
                        cols = st.columns(min(3, len(results["detections"])))

                        for i, detection in enumerate(results["detections"]):
                            with cols[i % 3]:
                                st.markdown(f"**Person {i+1}**")
                                emotion = detection.get("emotion", "Unknown")
                                confidence = detection.get("confidence", 0) * 100

                                # Get emoji for emotion
                                emoji = "😐"  # Default neutral
                                if emotion.lower() == "happy":
                                    emoji = "😀"
                                elif emotion.lower() == "sad":
                                    emoji = "😢"
                                elif emotion.lower() == "angry":
                                    emoji = "😠"
                                elif emotion.lower() == "surprised":
                                    emoji = "😲"
                                elif emotion.lower() == "fear":
                                    emoji = "😨"
                                elif emotion.lower() == "disgust":
                                    emoji = "🤢"

                                st.markdown(f"Emotion: {emotion} {emoji}")
                                st.progress(confidence / 100)
                                st.text(f"Confidence: {confidence:.1f}%")
                    else:
                        st.info("No faces detected in the image.")

    # Tab 2: Webcam Capture
    with tab2:
        st.header("Capture from Webcam")

        # Check if webcam is available
        if not st.checkbox("Enable Webcam", key="enable_webcam"):
            st.info("Click the checkbox to enable webcam capture.")
        else:
            # Create a placeholder for webcam feed
            webcam_placeholder = st.empty()

            try:
                # Open webcam
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error(
                        "Could not open webcam. Please check your camera settings."
                    )
                else:
                    # Capture frame
                    ret, frame = cap.read()
                    if ret:
                        # Convert from BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Display frame
                        webcam_placeholder.image(
                            frame_rgb, caption="Webcam Feed", use_column_width=True
                        )

                        # Capture and process buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Capture Image", key="capture_btn"):
                                # Store the captured frame
                                st.session_state.captured_frame = frame_rgb.copy()
                                st.success("Image captured!")

                        with col2:
                            if st.button(
                                "Process Captured Image", key="process_webcam_btn"
                            ):
                                if hasattr(st.session_state, "captured_frame"):
                                    # Process the captured frame
                                    results = process_image_with_api(
                                        st.session_state.captured_frame
                                    )

                                    if results:
                                        # Display the processed image with annotations
                                        if "processed_image" in results:
                                            processed_img = decode_base64_to_image(
                                                results["processed_image"]
                                            )
                                            st.image(
                                                processed_img,
                                                caption="Processed Image",
                                                use_column_width=True,
                                            )

                                        # Display detection results
                                        if (
                                            "detections" in results
                                            and results["detections"]
                                        ):
                                            st.subheader("Detected Emotions")

                                            # Create columns for each detection
                                            cols = st.columns(
                                                min(3, len(results["detections"]))
                                            )

                                            for i, detection in enumerate(
                                                results["detections"]
                                            ):
                                                with cols[i % 3]:
                                                    st.markdown(f"**Person {i+1}**")
                                                    emotion = detection.get(
                                                        "emotion", "Unknown"
                                                    )
                                                    confidence = (
                                                        detection.get("confidence", 0)
                                                        * 100
                                                    )

                                                    # Get emoji for emotion
                                                    emoji = "😐"  # Default neutral
                                                    if emotion.lower() == "happy":
                                                        emoji = "😀"
                                                    elif emotion.lower() == "sad":
                                                        emoji = "😢"
                                                    elif emotion.lower() == "angry":
                                                        emoji = "😠"
                                                    elif emotion.lower() == "surprised":
                                                        emoji = "😲"
                                                    elif emotion.lower() == "fear":
                                                        emoji = "😨"
                                                    elif emotion.lower() == "disgust":
                                                        emoji = "🤢"

                                                    st.markdown(
                                                        f"Emotion: {emotion} {emoji}"
                                                    )
                                                    st.progress(confidence / 100)
                                                    st.text(
                                                        f"Confidence: {confidence:.1f}%"
                                                    )
                                        else:
                                            st.info("No faces detected in the image.")
                                else:
                                    st.warning("Please capture an image first.")
                    else:
                        st.error("Failed to read from webcam.")

                    # Release webcam
                    cap.release()
            except Exception as e:
                st.error(f"Error accessing webcam: {str(e)}")

    # Tab 3: Real-time Detection
    with tab3:
        st.header("Real-time Emotion Detection")

        # Configuration options
        st.sidebar.subheader("Real-time Detection Settings")
        detection_frequency = st.sidebar.slider(
            "Detection Frequency (seconds)", 0.1, 2.0, 0.5, 0.1
        )
        show_fps = st.sidebar.checkbox("Show FPS", value=True)
        show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)

        # Start/stop button
        start_realtime = st.checkbox("Start Real-time Detection", key="start_realtime")

        if start_realtime:
            # Create placeholders
            video_placeholder = st.empty()
            stats_placeholder = st.empty()

            try:
                # Open webcam
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error(
                        "Could not open webcam. Please check your camera settings."
                    )
                    st.session_state.start_realtime = False
                else:
                    # Variables for FPS calculation
                    frame_count = 0
                    start_time = time.time()
                    last_detection_time = 0
                    current_detections = []

                    # Inform user
                    st.info(
                        "Real-time detection started. Click the checkbox again to stop."
                    )

                    # Loop until user stops
                    while st.session_state.start_realtime:
                        # Capture frame
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to read from webcam.")
                            break

                        # Convert from BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Update frame count for FPS
                        frame_count += 1
                        current_time = time.time()
                        elapsed_time = current_time - start_time

                        # Process frame with API at specified intervals
                        if current_time - last_detection_time >= detection_frequency:
                            # Process frame
                            results = process_video_frame_with_api(frame_rgb)

                            if results and "detections" in results:
                                current_detections = results["detections"]

                                # Get processed frame if available
                                if "processed_frame" in results:
                                    frame_rgb = decode_base64_to_image(
                                        results["processed_frame"]
                                    )
                                    frame_rgb = np.array(frame_rgb)

                            last_detection_time = current_time

                        # Draw detection results on frame if available
                        if current_detections:
                            # Create a copy to avoid modifying the original
                            display_frame = frame_rgb.copy()

                            for detection in current_detections:
                                if "bbox" in detection:
                                    # Get bounding box coordinates
                                    x, y, w, h = detection["bbox"]
                                    emotion = detection.get("emotion", "Unknown")
                                    confidence = detection.get("confidence", 0) * 100

                                    # Draw rectangle
                                    cv2.rectangle(
                                        display_frame,
                                        (x, y),
                                        (x + w, y + h),
                                        (0, 255, 0),
                                        2,
                                    )

                                    # Prepare label text

                                    label = f"{emotion}"
                                    if show_confidence:
                                        label += f": {confidence:.1f}%"

                                    # Get text size for better positioning
                                    text_size = cv2.getTextSize(
                                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                                    )[0]

                                    # Draw background for text
                                    cv2.rectangle(
                                        display_frame,
                                        (x, y - 25),
                                        (x + text_size[0], y),
                                        (0, 255, 0),
                                        -1,
                                    )

                                    # Draw text
                                    cv2.putText(
                                        display_frame,
                                        label,
                                        (x, y - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        (0, 0, 0),
                                        2,
                                    )
                        else:
                            display_frame = frame_rgb

                        # Show FPS if enabled
                        if show_fps and elapsed_time > 0:
                            fps = frame_count / elapsed_time
                            cv2.putText(
                                display_frame,
                                f"FPS: {fps:.1f}",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2,
                            )

                        # Display frame
                        video_placeholder.image(
                            display_frame,
                            caption="Real-time Detection",
                            use_column_width=True,
                        )

                        # Display statistics
                        if current_detections:
                            emotions_detected = [
                                det.get("emotion", "Unknown")
                                for det in current_detections
                            ]
                            emotion_counts = {
                                emotion: emotions_detected.count(emotion)
                                for emotion in set(emotions_detected)
                            }

                            stats_text = "Detected Emotions:\n"
                            for emotion, count in emotion_counts.items():
                                # Get emoji for emotion
                                emoji = "😐"  # Default neutral
                                if emotion.lower() == "happy":
                                    emoji = "😀"
                                elif emotion.lower() == "sad":
                                    emoji = "😢"
                                elif emotion.lower() == "angry":
                                    emoji = "😠"
                                elif emotion.lower() == "surprised":
                                    emoji = "😲"
                                elif emotion.lower() == "fear":
                                    emoji = "😨"
                                elif emotion.lower() == "disgust":
                                    emoji = "🤢"

                                stats_text += f"{emotion} {emoji}: {count} "

                            stats_placeholder.text(stats_text)

                        # Reset FPS calculation after 5 seconds
                        if elapsed_time > 5:
                            start_time = time.time()
                            frame_count = 0

                        # Add small delay to reduce CPU usage
                        time.sleep(0.01)

                    # Release webcam when stopped
                    cap.release()
            except Exception as e:
                st.error(f"Error in real-time detection: {str(e)}")
                st.session_state.start_realtime = False

    # Add settings in sidebar
    st.sidebar.subheader("Application Settings")

    # API configuration
    st.sidebar.text_input(
        "API URL", value=API_URL, key="api_url", help="URL of the emotion detection API"
    )

    if st.sidebar.button("Update API URL"):
        # Update API URL from input
        global API_URL
        API_URL = st.session_state.api_url
        st.sidebar.success("API URL updated!")

    # Advanced settings expander
    with st.sidebar.expander("Advanced Settings"):
        st.slider(
            "Image Quality",
            50,
            100,
            85,
            5,
            help="Quality of images sent to API (lower means smaller size)",
        )
        st.checkbox(
            "Enable Face Tracking",
            value=True,
            help="Track faces across video frames for smoother detection",
        )
        st.slider(
            "Detection Confidence Threshold",
            0.0,
            1.0,
            0.5,
            0.05,
            help="Minimum confidence threshold for emotion detection",
        )

    # App info in sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Emotion Detection App v1.0")
    st.sidebar.caption("© 2025 Your Company")


if __name__ == "__main__":
    main()
