import streamlit as st
from contextlib import contextmanager
from utils import load_image_from_bytes, to_gray, draw_keypoints, draw_loftr_keypoints
from detector import *
from deep import *
import matplotlib.pyplot as plt
import time

st.title("Image Keypoint Detection")

detector_options = ["SIFT", "FAST", "BRIEF", "ORB", "MSER", "AKAZE", "BRISK"]
deep_matcher_options = ["None", "LoFTR"]

st.sidebar.title("Options")
selected_detector = st.sidebar.selectbox("Choose a detector", detector_options)
selected_deep_matcher = st.sidebar.selectbox("Choose a deep matcher", deep_matcher_options)

detectors = {
    "SIFT": SIFTDetector(),
    "FAST": FastDetector(),
    "BRIEF": BRIEFDetector(),
    "ORB": ORBDetector(),
    "MSER": MSERDetector(),
    "AKAZE": AKAZEDetector(),
    "BRISK": BRISKDetector(),
    "LoFTR": LoFTRMatcher()
}

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'counter' not in st.session_state:
    st.session_state.counter = 0

def update_history(detector_name, keypoints, processing_time):
    st.session_state.counter += 1
    unique_id = f"{detector_name} #{st.session_state.counter}"
    
    processing_time = float(processing_time.split(' ')[0])

    history = st.session_state['history']
    if len(history) >= 10:
        history.pop(0)
    history.append({
        "Detector": unique_id,
        "Keypoints": keypoints,
        "Processing Time": processing_time
    })

    st.session_state['history'] = history

def plot_combined_chart():
    history = st.session_state['history']
    if not history:
        st.write("No data to display yet.")
        return

    fig, ax1 = plt.subplots()

    detectors = [record["Detector"] for record in history]
    keypoints = [record["Keypoints"] for record in history]
    processing_times = [record["Processing Time"] for record in history]

    ax1.bar(detectors, processing_times, color='b', label='Processing Time (ms)')
    ax1.set_xlabel('Detector')
    ax1.set_ylabel('Processing Time (ms)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(detectors, keypoints, color='g', marker='o', label='Average Keypoints')
    ax2.set_ylabel('Count', color='k')
    ax2.tick_params(axis='y', labelcolor='k')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    st.pyplot(fig)

timing_results = {}

@contextmanager
def timer(label):
    start = time.time()
    yield
    end = time.time()
    timing_results[label] = f"{(end - start) * 1000:.2f} ms"

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if uploaded_file is not None:

    image = load_image_from_bytes(uploaded_file.getvalue())
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    gray_image = to_gray(image)

    with timer("Detection Process"):
        if selected_deep_matcher == "None":
            detector_name = selected_detector 
            detector = detectors[selected_detector]
            keypoints = detector.detect(gray_image)
            keypoints, _ = detector.compute(gray_image, keypoints)
            keypoints_image = draw_keypoints(image, keypoints)
        else:
            detector_name = selected_deep_matcher
            detector = detectors[selected_deep_matcher]
            scale = 1000 / max(image.shape[:2])
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
            gray1 = to_gray(image)
            keypoints = detector.detect_keypoints(gray1)
            keypoints_image = draw_loftr_keypoints(image, keypoints)

    update_history(detector_name, len(keypoints), timing_results["Detection Process"])

    st.image(keypoints_image, caption="Detected Keypoints", use_column_width=True)

    timing_data = [{"Process Type": key, "Time To Take (Miliseconds)": value} for key, value in timing_results.items()]
    st.sidebar.table(timing_data)

    st.sidebar.metric(label="Number of Keypoints", value=len(keypoints))

    plot_combined_chart()