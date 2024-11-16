# Breast Cancer Diagnosis Predictor
## Overview
The Breast Cancer Diagnosis Predictor app is a machine learning-powered tool designed to assist in diagnosing breast cancer. The app predicts whether a breast mass is benign or malignant based on various measurements. It uses a public dataset (Breast Cancer Wisconsin Diagnostic Data Set) for training and prediction. The app allows users to interactively input these measurements through sliders and visualizes the prediction results along with probabilities, as well as providing a radar chart comparing different input values.

The app was built for educational purposes, demonstrating the use of machine learning techniques and providing a tool for predicting breast cancer diagnoses. This tool is not intended for professional medical use.

A live version of the application is hosted on Streamlit Community Cloud.

Features
Interactive Input Sliders: Users can adjust various measurement values (e.g., radius, texture, area) and see how they affect the diagnosis.
Radar Chart Visualization: A radar chart displays the input values for different features, allowing for a clear comparison between benign and malignant cases.
Machine Learning Prediction: The app uses a trained classification model (e.g., Random Forest or Logistic Regression) to predict whether the tumor is benign or malignant based on user input.
Predicted Probabilities: The app shows the probabilities of the mass being benign or malignant.
Styled Diagnosis Output: The prediction (benign or malignant) is displayed with custom CSS styling for better visual appeal.
Requirements
Python 3.7+
Streamlit
Plotly
Scikit-learn
Pandas
Numpy
Installation
To run this app, follow the steps below to set up your environment:

1. Create a virtual environment:
You can use conda to create a virtual environment for this project:

bash
Copier le code
conda create -n breast-cancer-diagnosis python=3.10
conda activate breast-cancer-diagnosis
Alternatively, you can use virtualenv or your preferred environment manager.

2. Install dependencies:
Install the necessary packages by running the following command:

bash
Copier le code
pip install -r requirements.txt
This will install all required dependencies, including Streamlit, Plotly, and scikit-learn.

Usage
Run the app:

To start the app, run the following command in the terminal:

bash
Copier le code
streamlit run app.py
This will launch the app in your default web browser.

Interacting with the App:

Adjust the sliders in the sidebar to input various measurements such as radius, texture, perimeter, and area.
Once the measurements are set, the app will display the predicted diagnosis (benign or malignant) along with the associated probabilities.
The radar chart will visualize the comparison between the input measurements across different categories.
Viewing the results:

The app will show:

A styled diagnosis message indicating whether the mass is benign or malignant.
The probabilities for each class (benign and malignant).
A radar chart comparing the input measurements across different categories.
File Structure
plaintext
Copier le code
📂 breast-cancer-diagnosis-predictor
├── app.py                 # Main app code
├── model.pkl              # Trained machine learning model
├── requirements.txt       # Required dependencies
├── styles.css             # Custom CSS for styling the app
└── README.md              # Project documentation
Customization
To change the CSS styling, modify the styles.css file. You can adjust the colors and fonts used for the prediction display.
Replace the model.pkl with a new trained machine learning model if you wish to experiment with different models or update the existing one.
Contributing
Contributions are welcome! If you would like to contribute, please fork the repository, make changes in a new branch, and submit a pull request. Ensure that you follow the repository's style guidelines and include tests where appropriate.

Acknowledgments
This project uses:

Streamlit: A powerful framework for building data-driven applications.
scikit-learn: For machine learning tools used to train the classification model.
Plotly: For creating interactive radar charts.
License
This project is licensed under the MIT License. See the LICENSE file for details.
