# Pneumonia Detection Using Machine Learning Techniques and Explainable AI

This project is a Streamlit-based application that detects pneumonia using machine learning techniques and visualizes the results with Grad-CAM heatmaps for enhanced explainability. It also generates a detailed multi-page PDF report for each prediction.

## Features

1. **Pneumonia Detection**:

   - Supports both chest X-ray and CT scan images.
   - Utilizes pre-trained DenseNet121 models.

2. **Explainable AI**:

   - Grad-CAM heatmaps to visualize regions of interest.
   - Severity grading based on heatmap analysis.

3. **Comprehensive Report**:

   - Generates a multi-page PDF report including:
     - Detection results and confidence.
     - Severity level (if pneumonia is detected).
     - Explanations and visualizations (original image and Grad-CAM heatmap).

## Prerequisites

1. Python 3.8 or higher.

2. The following Python packages:

   - `streamlit`
   - `tensorflow`
   - `numpy`
   - `cv2` (OpenCV)
   - `Pillow`
   - `reportlab`

3. Datasets used:

   - [Chest X-ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
   - [CT Scan Dataset](https://www.kaggle.com/datasets/anaselmasry/covid19normalpneumonia-ct-images)

## Running the Application

1. Start the Streamlit server:

   ```bash
   streamlit run Pneumonia_Detection_App.py
   ```

2. Open the URL displayed in the terminal (default is `http://localhost:8501/`) in a web browser.

## Usage

1. **Select Image Type**:

   - Use the dropdown menu to select the image type: `X-ray` or `CT-scan`.

2. **Upload Image**:

   - Upload a valid image file (JPG, JPEG, or PNG format).

3. **Detect Pneumonia**:

   - Click the "Detect" button to analyze the image.
   - View the detection result, confidence, and Grad-CAM visualization (if applicable).

4. **Download Report**:

   - Click the "Download Detailed Report" button to download the PDF report.

## Explanation of Severity Grading

1. Grad-CAM highlights regions of interest (areas with high opacity and consolidation).
2. Severity is graded based on the percentage of lung area affected:
   - Mild: < 20%
   - Moderate: 20% - 50%
   - Severe: > 50%

## Project Workflow

1. **Input Image Preprocessing**:

   - Images are resized to 224x224 and normalized.

2. **Prediction**:

   - Model predicts the probability of pneumonia.

3. **Grad-CAM Visualization**:

   - Highlights key regions contributing to the prediction.

4. **Severity Calculation**:

   - Based on the intensity and extent of highlighted areas.

5. **PDF Report Generation**:

   - Summarizes detection results, severity, and includes visualizations.

