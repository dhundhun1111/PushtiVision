import streamlit as st
import time
import cv2
import torch
import numpy as np
import random
from PIL import Image
from ultralytics import YOLO

# Load YOLO Model
@st.cache_resource()
def load_model():
    model_url = "https://huggingface.co/dhundhun1111/PushtiVision/resolve/main/yolov8x.pt"
    model_path = torch.hub.load_state_dict_from_url(model_url, map_location=torch.device('cpu'))
    model = YOLO(model_path)
    return model

# Perform Inference
def predict(image, model):
    img = np.array(image)
    results = model.predict(img, conf=0.25, iou=0.45)
    detections = results[0].boxes.data.cpu().numpy()
    return detections

# Food Calorie Mapping
food_calories = {
    "apple": 52,
    "banana": 89,
    "burger": 354,
    "pizza": 266,
    "sandwich": 250,
    "salad": 150,
    "fries": 312,
    "sushi": 300,
}

# Streamlit UI Customization
st.set_page_config(page_title="NutriVision Inference", layout="centered")
st.markdown("""
    <style>
        body {
            background-color: #f8e1e1;  /* Light red background */
            color: #9e2a2f;  /* Dark red text */
        }
        .stButton>button {
            background-color: #9e2a2f;  /* Dark red button */
            color: white;
        }
        .stButton>button:hover {
            background-color: #7c1d1d;  /* Darker red on hover */
        }
        .stFileUploader>label {
            background-color: #ffccd5;  /* Light pink background */
            color: #9e2a2f;  /* Dark red text */
        }
        .stTextInput>label {
            color: #9e2a2f;  /* Dark red text */
        }
        .stTitle {
            color: #9e2a2f;  /* Dark red text for title */
        }
        /* CSS for revolving and rotating emojis */
        @keyframes orbit {
            0% {
                transform: rotate(0deg) translateX(120px) rotate(0deg);
            }
            100% {
                transform: rotate(360deg) translateX(120px) rotate(-360deg);
            }
        }
        .loading-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 400px;
            position: relative;
        }
        .loading-emoji {
            font-size: 3rem;
            position: absolute;
            animation: orbit 11s infinite linear;
        }
        .loading-emoji:nth-child(1) {
            animation-delay: 0s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(2) {
            animation-delay: 1s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(3) {
            animation-delay: 2s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(4) {
            animation-delay: 3s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(5) {
            animation-delay: 4s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(6) {
            animation-delay: 5s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(7) {
            animation-delay: 6s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(8) {
            animation-delay: 7s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(9) {
            animation-delay: 8s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(9) {
            animation-delay: 8s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(9) {
            animation-delay: 8s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(10) {
            animation-delay: 9s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(11) {
            animation-delay: 10s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(12) {
            animation-delay: 11s;
            top: 50%;
            left: 50%;
        }
        
        
    </style>
""", unsafe_allow_html=True)
# -------------------------------
# Function to Display the Loading Screen
# -------------------------------
def show_loading_screen():
    loading_container = st.empty()
    
    # Show the "Loading..." message
    loading_container.markdown("<h3 style='color:#9e2a2f;'>🔄 Loading... Please wait!</h3>", unsafe_allow_html=True)

    # Create the rotating emoji effect
    with loading_container.container():
        loading_container.markdown("""
        <div class="loading-container">
            <span class="loading-emoji">🍎</span>
            <span class="loading-emoji">🍕</span>
            <span class="loading-emoji">🥫</span>
            <span class="loading-emoji">🍔</span>
            <span class="loading-emoji">🍊</span>
            <span class="loading-emoji">🥛</span>
            <span class="loading-emoji">🥚</span>
            <span class="loading-emoji">🐟</span>
            <span class="loading-emoji">🍗</span>
            <span class="loading-emoji">🌭</span>
            <span class="loading-emoji">🍲</span>
        </div>

        """, unsafe_allow_html=True)
    
    # Wait for 5 seconds before loading the main content
    time.sleep(12)
    
    # Remove loading container after the delay
    loading_container.empty()


# Main App Logic
def main_app():
    st.title("PusthiVision: Nutrition Estimator")
    st.write("Upload a food image and get estimated nutritional details.")
    
    motivational_quotes = [
        "Fuel Your Body with the Right Nutrients!",
        "Stay Fit, Eat Well!",
        "Your Body Deserves the Best—Give It Nutrition!",
        "A Healthy Outside Starts from the Inside!",
        "Good Food, Good Mood!",
    ]
    quote = random.choice(motivational_quotes)
    st.markdown(f"<h3 style='color:#9e2a2f;'>💪 {quote} 💪</h3>", unsafe_allow_html=True)
    
    model = load_model()
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("Detecting food items..."):
            detections = predict(image, model)
        
        if len(detections) > 0:
            total_calories = 0
            st.write("## Detection Results")
            for det in detections:
                cls_id = int(det[5])
                conf = det[4]
                class_name = model.names[cls_id]
                calories = food_calories.get(class_name, "Unknown")
                total_calories += calories if isinstance(calories, int) else 0
                st.write(f"🍽 **{class_name}** (Confidence: {conf:.2f}) - Estimated Calories: {calories}")
            st.write(f"### 🔥 Total Estimated Calories: {total_calories} kcal")
        else:
            st.write("❌ No food items detected.")

# Run the App
show_loading_screen()
main_app()
