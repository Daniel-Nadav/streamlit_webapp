import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- Configuration & Sidebar ---
st.set_page_config(page_title="Image Segmentation App", layout="centered")

st.sidebar.title("Settings")

# Method Selection
segmentation_method = st.sidebar.selectbox(
    "Choose Segmentation Method",
    ("K-Means Clustering", "Otsu's Thresholding")
)

# Dynamic Sidebar controls based on selection
params = {}
if segmentation_method == "K-Means Clustering":
    params['k_value'] = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
    params['attempts'] = st.sidebar.slider("Attempts", 1, 20, 10)
elif segmentation_method == "Otsu's Thresholding":
    st.sidebar.info("Otsu's method automatically finds the optimal threshold value to separate foreground from background.")
    params['blur'] = st.sidebar.slider("Gaussian Blur Kernel Size (odd numbers only)", 1, 21, 5, step=2)

st.title("🖼️ Image Segmentation App")
st.markdown("Upload an image to separate objects from the background using different algorithms.")

# --- Helper Functions ---

def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img)

def segment_kmeans(img, k, attempts):
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
    
    # Create visual representation
    segmented_image = centers[labels]
    segmented_image = segmented_image.reshape(img.shape)
    
    return segmented_image, labels, centers

def segment_otsu(img, blur_kernel):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian Blur to reduce noise (improves Otsu results)
    if blur_kernel > 1:
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        
    # Apply Otsu's Thresholding
    # thresh_val is the optimal threshold found, mask is the binary image
    thresh_val, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Create a segmented color image using the mask
    # Bitwise AND requires mask to be uint8
    result = cv2.bitwise_and(img, img, mask=mask)
    
    return result, mask, thresh_val

# --- Main App Logic ---

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_image = load_image(uploaded_file)
    
    # Display Original
    st.subheader("Original Image")
    st.image(original_image, use_column_width=True)
    
    st.divider()

    # --- K-MEANS LOGIC ---
    if segmentation_method == "K-Means Clustering":
        st.subheader("K-Means Results")
        
        with st.spinner(f"Clustering with k={params['k_value']}..."):
            segmented_img, labels, centers = segment_kmeans(
                original_image, 
                params['k_value'], 
                params['attempts']
            )
        
        st.image(segmented_img, caption="Segmented Image (All Clusters)", use_column_width=True)
        
        st.markdown("#### Isolate Specific Cluster")
        selected_cluster = st.radio(
            "Select a cluster index to visualize:",
            options=range(params['k_value']),
            horizontal=True
        )
        
        # Create mask for specific cluster
        # Reshape labels to match image height/width
        mask = labels.reshape(original_image.shape[:2]) == selected_cluster
        
        # Apply mask to original image
        isolated_object = original_image.copy()
        isolated_object[~mask] = [0, 0, 0] # Black out everything else
        
        st.image(isolated_object, caption=f"Cluster {selected_cluster} Isolated", use_column_width=True)

    # --- OTSU LOGIC ---
    elif segmentation_method == "Otsu's Thresholding":
        st.subheader("Otsu's Binarization Results")
        
        result, mask, thresh_val = segment_otsu(original_image, params['blur'])
        
        st.info(f"Optimal Threshold Value Calculated: {thresh_val}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(mask, caption="Binary Mask (Black/White)", use_column_width=True)
        
        with col2:
            st.image(result, caption="Segmented Result (Mask Applied)", use_column_width=True)
            
        st.caption("*Note: Otsu works best when there is a high contrast (bi-modal histogram) between foreground and background.*")
