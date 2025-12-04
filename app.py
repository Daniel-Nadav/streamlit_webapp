import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- Configuration & Sidebar ---
st.set_page_config(page_title="Object Area Calculator", layout="centered")

st.sidebar.title("Settings")
marker_size = st.sidebar.number_input("AruCo Marker Side Length (cm)", min_value=1.0, value=5.0, step=0.1)
k_value = st.sidebar.slider("K-Means Clusters (k)", min_value=2, max_value=10, value=3)
attempts = st.sidebar.slider("K-Means Attempts", min_value=1, max_value=20, value=10)

st.title("📏 Object Area Calculator")
st.markdown(
    """
    **Instructions:**
    1. Upload an image containing an **AruCo Marker** (DICT_5X5_50) and the object you want to measure.
    2. The app will detect the marker to establish a scale.
    3. It will segment the image using K-Means clustering.
    4. Select the specific cluster corresponding to your object to see its real-world area.
    """
)

# --- Helper Functions ---

def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img)

def detect_aruco(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Define Dictionary and Parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    parameters = cv2.aruco.DetectorParameters()
    
    # Create Detector
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    # Detect
    corners, ids, rejected = detector.detectMarkers(gray)
    
    return corners, ids

def segment_image_kmeans(img, k=3, attempts=10):
    # Reshape to 2D array of pixels
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Define criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # Perform K-Means
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

    # Convert centers back to uint8
    centers = np.uint8(centers)
    
    # Flatten labels
    labels = labels.flatten()
    
    # Create segmented image
    segmented_image = centers[labels]
    segmented_image = segmented_image.reshape(img.shape)
    
    return segmented_image, labels, centers

# --- Main App Logic ---

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Load and Display Original
    original_image = load_image(uploaded_file)
    
    # Create a copy for visualization
    display_image = original_image.copy()
    
    st.subheader("1. ArUco Detection")
    
    # 2. Detect ArUco
    corners, ids = detect_aruco(original_image)
    
    if corners:
        # Draw marker
        int_corners = np.int32(corners)
        cv2.polylines(display_image, int_corners, True, (0, 255, 0), 5)
        
        # Calculate Pixel Area of Marker
        aruco_area_px = cv2.contourArea(corners[0])
        
        # Calculate Ratio
        aruco_area_cm = marker_size * marker_size
        cm2_per_px2 = aruco_area_cm / aruco_area_px
        
        st.image(display_image, caption="Detected Marker", use_column_width=True)
        st.success(f"Marker Detected! Ratio: {cm2_per_px2:.6f} cm²/px")
        
        # 3. Segmentation
        st.subheader("2. Image Segmentation")
        
        with st.spinner("Performing K-Means clustering..."):
            segmented_img, labels, centers = segment_image_kmeans(original_image, k=k_value, attempts=attempts)
        
        st.image(segmented_img, caption=f"Segmented Image (k={k_value})", use_column_width=True)
        
        # 4. Area Calculation Selection
        st.subheader("3. Select Object Cluster")
        
        # Display color choices for the user
        cols = st.columns(k_value)
        
        selected_cluster = st.radio(
            "Which color represents your object?",
            options=range(k_value),
            format_func=lambda x: f"Cluster {x}"
        )
        
        # Visualize the chosen cluster only
        # Mask: True where label matches selected cluster
        mask = labels.reshape(original_image.shape[:2]) == selected_cluster
        
        # Create a visualization where only the selected object is shown, others black
        isolated_object = original_image.copy()
        isolated_object[~mask] = [0, 0, 0]
        
        st.image(isolated_object, caption=f"Isolated Object (Cluster {selected_cluster})", use_column_width=True)
        
        # Count pixels
        object_px_count = np.sum(mask)
        real_area = object_px_count * cm2_per_px2
        
        st.markdown("### 📊 Results")
        col1, col2 = st.columns(2)
        col1.metric("Pixel Count", f"{object_px_count} px")
        col2.metric("Real Area", f"{real_area:.2f} cm²")

    else:
        st.error("No ArUco marker detected. Please upload an image with a visible DICT_5X5_50 marker.")
        st.image(original_image, caption="Original Image", use_column_width=True)
