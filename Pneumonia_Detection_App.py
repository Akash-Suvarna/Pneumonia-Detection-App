import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# Function to load and preprocess images for prediction
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Grad-CAM visualization function
def generate_grad_cam(img_path, model, selected_layer_name):
    img_array = load_and_preprocess_image(img_path)

    # Predict the class
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds[0])

    # Grad-CAM process
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(selected_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = np.maximum(np.sum(weights * conv_outputs, axis=-1), 0)
    heatmap /= np.max(heatmap)

    # Resize and superimpose heatmap
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    scaled_heatmap = np.uint8(255 * heatmap_resized)  # Scale heatmap to [0, 255]
    heatmap_img = cv2.applyColorMap(scaled_heatmap, cv2.COLORMAP_JET)

    # Swap red and blue channels
    heatmap_img[:, :, [0, 2]] = heatmap_img[:, :, [2, 0]]  # Swap B (blue) and R (red)

    # Blend the heatmap with the original image
    original_img = np.uint8(255 * img_array[0])  # Scale to [0, 255]
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_img, 0.4, 0)  # Adjust blend ratio

    return img_array[0], superimposed_img, predicted_class, preds[0], heatmap_img, heatmap_resized


# Function to create and download a multi-page PDF report
def create_multi_page_pdf(report_file, original_image, grad_cam_image, detection_result, severity, confidence, grad_cam_explanation, severity_explanation):
    c = canvas.Canvas(report_file, pagesize=letter)
    width, height = letter
    margin = 50
    line_height = 20
    max_lines_per_page = int((height - 2 * margin) / line_height)  # Calculate maximum lines per page
    current_y = height - margin

    def add_new_page():
        """Helper function to start a new page and reset the cursor."""
        nonlocal current_y
        c.showPage()
        current_y = height - margin

    # Add title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, current_y, "Pneumonia Detection Report")
    current_y -= 2 * line_height

    # Add detection result and confidence
    c.setFont("Helvetica", 12)
    c.drawString(margin, current_y, f"Detection Result: {detection_result}")
    current_y -= line_height
    c.drawString(margin, current_y, f"Confidence: {confidence * 100:.2f}%")
    current_y -= 2 * line_height

    # Add severity (if applicable)
    if detection_result == "Pneumonia Detected!":
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, current_y, f"Severity: {severity}")
        current_y -= 2 * line_height

        # Add Grad-CAM explanation
        c.setFont("Helvetica", 10)
        c.drawString(margin, current_y, "Why These Areas Are Highlighted:")
        current_y -= line_height
        for line in grad_cam_explanation:
            if current_y < margin:  # Start a new page if there's not enough space
                add_new_page()
            c.drawString(margin, current_y, f"- {line}")
            current_y -= line_height

        # Add severity explanation
        c.setFont("Helvetica", 10)
        if current_y < margin:
            add_new_page()
        c.drawString(margin, current_y, "Severity Explanation:")
        current_y -= line_height
        for line in severity_explanation:
            if current_y < margin:
                add_new_page()
            c.drawString(margin, current_y, f"- {line}")
            current_y -= line_height

    # Add original image and Grad-CAM heatmap
    if current_y < margin + 300:  # Ensure enough space for images
        add_new_page()
    original_image_path = "original_image.jpg"
    Image.fromarray((original_image * 255).astype(np.uint8)).save(original_image_path)
    c.drawString(margin, current_y, "Original Image:")
    c.drawImage(original_image_path, margin, current_y - 300, width=200, height=200)
    current_y -= 320

    if detection_result == "Pneumonia Detected!":
        grad_cam_image_path = "grad_cam_image.jpg"
        Image.fromarray(grad_cam_image).save(grad_cam_image_path)
        if current_y < margin + 300:
            add_new_page()
        c.drawString(margin, current_y, "Grad-CAM Heatmap:")
        c.drawImage(grad_cam_image_path, margin, current_y - 300, width=200, height=200)
        current_y -= 320

    c.save()


# Streamlit UI for Pneumonia Detection App
st.title("Pneumonia Detection and CAM Visualization")
st.markdown("Upload an image and select its type (X-ray or CT-scan) to detect pneumonia and visualize the results.")

# Session state to reset on dropdown change
if "last_selected" not in st.session_state:
    st.session_state.last_selected = "X-ray"


def reset_app():
    st.session_state.last_selected = image_type
    st.session_state.uploaded_file = None
    st.session_state.result_shown = False


# Dropdown for selecting the image type
image_type = st.selectbox("Select Image Type", ["X-ray", "CT-scan"], on_change=reset_app)

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Load the appropriate model based on image type
    if image_type == "X-ray":
        model = tf.keras.models.load_model("chest_xray_model.keras")
    else:
        model = tf.keras.models.load_model("ct_model_final.keras")

    # Button to perform detection
    if st.button("Detect"):
        # Save the uploaded image locally
        img_path = uploaded_file.name
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        img_array = load_and_preprocess_image(img_path)
        preds = model.predict(img_array)
        pred_class = np.argmax(preds[0])

        # Display pneumonia detection result in the center
        result_text = "Pneumonia Detected!" if pred_class == 1 else "Normal!"
        result_color = "red" if pred_class == 1 else "green"
        confidence = preds[0][1] if pred_class == 1 else preds[0][0]

        st.markdown(
            f"<h2 style='text-align:center; color:{result_color}; font-weight:bold;'>{result_text}</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='text-align:center; font-size:18px;'>Confidence: {confidence * 100:.2f}%</p>",
            unsafe_allow_html=True,
        )

        if pred_class == 1:
            # Generate Grad-CAM heatmap only for pneumonia
            _, grad_cam_img, _, _, heatmap_img, heatmap_resized = generate_grad_cam(img_path, model, "conv5_block16_2_conv")
            st.image(grad_cam_img, caption="CAM Heatmap", use_column_width=True)

            # Severity Grading
            heatmap_area = np.sum(heatmap_resized > 0.5)  # Areas with intensity above 0.5
            total_area = heatmap_resized.size
            severity_ratio = heatmap_area / total_area

            if severity_ratio < 0.2:
                severity = "Mild"
            elif severity_ratio < 0.5:
                severity = "Moderate"
            else:
                severity = "Severe"

            # Explanations
            grad_cam_explanation = [
                "The model is focusing on areas with high opacity and consolidation.",
                "These regions suggest lung inflammation and tissue damage."
            ]
            severity_explanation = [
                "Severity is based on affected areas in the heatmap.",
                f"Approximately {severity_ratio * 100:.2f}% of the lung area shows pneumonia signs."
            ]

            # Create and download report
            report_file = "pneumonia_report.pdf"
            create_multi_page_pdf(report_file, img_array[0], grad_cam_img, result_text, severity, confidence, grad_cam_explanation, severity_explanation)

        else:
            # For normal case
            report_file = "normal_report.pdf"
            create_multi_page_pdf(report_file, img_array[0], None, "Normal", "", confidence, [], [])

        # Provide download button for the PDF
        with open(report_file, "rb") as f:
            st.download_button(
                label="Download Detailed Report",
                data=f,
                file_name=report_file,
                mime="application/pdf",
            )
