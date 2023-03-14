import streamlit as st
import cv2
import numpy as np
from PIL import Image
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests
import torch

# Set page title, icon, and layout
st.set_page_config(page_title='Medical Image Segmentation', page_icon=':microscope:', layout='wide')

# Set page header style
header_style = """
    <style>
        .stApp {
            background-color: #f2f2f2;
        }
        .stApp header {
            background-color: #28a745;
            color: #fff;
            font-family: 'Times New Roman', Times, serif;
            font-size: 36px;
            font-weight: bold;
            padding: 0.5rem;
            text-align: center;
            text-shadow: 1px 1px #333;
        }
        .stApp .stMarkdown {
            color: #000;
            font-family: 'Times New Roman', Times, serif;
            font-size: 24px;
            font-weight: bold;
            padding: 1rem;
            text-align: center;
            text-shadow: 1px 1px #eee;
        }
    </style>
"""

# Add the header style to the app
st.markdown(header_style, unsafe_allow_html=True)


@st.cache_data
def segment_image(img):
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    inputs = feature_extractor(images=img, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

    # Get the predicted class for each pixel
    segmented_img = torch.argmax(outputs.logits.squeeze(), dim=0).cpu().numpy()

    return segmented_img


def main():
    # Set page header
    st.title('Medical Image Segmentation')

    # Add file uploader
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

    # Segment the uploaded image and display the result
    if uploaded_file is not None:
        # Read the uploaded image
        img = Image.open(uploaded_file)


        img_scaled = cv2.resize(np.array(img), (512, 512))

        # Segment the image
        segmented_img = segment_image(img_scaled)

        # Display the original and segmented images
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Original Image')
            st.image(img, use_column_width=True)

        with col2:
            st.subheader('Segmented Image')
            st.image(segmented_img, use_column_width=True)


if __name__ == '__main__':
    main()
