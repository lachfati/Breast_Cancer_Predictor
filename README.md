# **Breast Cancer Diagnosis Predictor**

## **Overview**

The **Breast Cancer Diagnosis Predictor** app is a machine learning-powered tool designed to assist in diagnosing breast cancer. The app predicts whether a breast mass is **benign** or **malignant** based on various measurements. It uses a public dataset (Breast Cancer Wisconsin Diagnostic Data Set) for training and prediction. The app allows users to interactively input these measurements through sliders and visualizes the prediction results along with probabilities, as well as providing a radar chart comparing different input values.

The app was built for educational purposes, demonstrating the use of machine learning techniques and providing a tool for predicting breast cancer diagnoses. This tool is not intended for professional medical use.

A live version of the application is hosted on [Streamlit Community Cloud](https://breast-cancer-predictor-24.streamlit.app/).

---

## **Features**

- **Interactive Input Sliders:** Users can adjust various measurement values (e.g., radius, texture, area) and see how they affect the diagnosis.
- **Radar Chart Visualization:** A radar chart displays the input values for different features, allowing for a clear comparison between benign and malignant cases.
- **Machine Learning Prediction:** The app uses a trained classification model (e.g., Random Forest or Logistic Regression) to predict whether the tumor is benign or malignant based on user input.
- **Predicted Probabilities:** The app shows the probabilities of the mass being benign or malignant.
- **Styled Diagnosis Output:** The prediction (benign or malignant) is displayed with custom CSS styling for better visual appeal.

---

## **Requirements**

- Python 3.7+
- Streamlit
- Plotly
- Scikit-learn
- Pandas
- Numpy

---

## **Installation**

To run this app, follow the steps below to set up your environment:

### **1. Create a virtual environment:**

**To create a virtual environment for this project:**

```bash
python -m venv venv
```
**Activating/Deactivating a Python Virtual Environment:**
Windows : 
```bash
venv/Scripts/activate
deacitvate
```
**To install the required packages, run:**

```bash
pip install -r requirements.txt
```

This will install all the necessary dependencies, including Streamlit, OpenCV, and scikit-image.

## Usage
To start the app, simply run the following command:

```bash
streamlit run app/main.py
```

This will launch the app in your default web browser. You can then upload an image of cells to analyze and adjust the various settings to customize the analysis. Once you are satisfied with the results, you can export the measurements to a CSV file for further analysis.
