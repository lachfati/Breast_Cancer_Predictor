import streamlit as st 
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go

#from model.main import get_clean_data

def get_clean_data():
    data = pd.read_csv('data/data.csv')
    data = data.drop(['Unnamed: 32', 'id'], axis=1)  # Correction de la suppression de colonnes
    data['diagnosis'] = data['diagnosis'].astype('category').cat.codes
    return data

def add_sidebar():
    st.sidebar.header('Cell Nuclei Measrument')
    data = get_clean_data()
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]
    input_dict = {}
    for label, value in slider_labels :
        input_dict[value] = st.sidebar.slider(
            label,
            min_value = float(0),
            max_value = float(data[value].max()),
            value = float(data[value].mean())
        )
    return input_dict

def get_scaled_values(input_dict):
  data = get_clean_data()
  
  X = data.drop(['diagnosis'], axis=1)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict

def get_radar_chart(input):

    input = get_scaled_values(input)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

    fig = go.Figure()
    mean_inputs = []
    se_inputs = []
    worst_input = []

    for key, value in input.items():
        # Select  "mean" input
        if "mean" in key:
            mean_inputs.append(value)

        # Select "se" input
        elif "se" in key:
            se_inputs.append(value)

        # Select "worst" input
        elif "worst" in key:
            worst_input.append(value)
            
    # add trace for "Mean"
    fig.add_trace(go.Scatterpolar(
        r=mean_inputs,
        theta=categories,
        fill='toself',
        name='Mean'
    ))
    
    # add trace for "Standard Error" (SE)
    fig.add_trace(go.Scatterpolar(
        r=se_inputs,
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    # add trace for "Worst"
    fig.add_trace(go.Scatterpolar(
        r=worst_input,
        theta=categories,
        fill='toself',
        name='Worst'
    ))

    # Page congiguration
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1] 
            )
        ),
        showlegend=True
    )
    return fig

def add_prediction(input_data):

    model = pickle.load(open('model/model.pkl', 'rb'))
    scaler = pickle.load(open('model/scaler.pkl', 'rb'))

    #Convert the input_data wich is a dict to numpy array
    input_array = np.array(list(input_data.values())).reshape(1,-1) #the reshape is to a have the data in 1 line

    #Scaling the input data using the same scaler of trained data
    input_scaler = scaler.transform(input_array)

    prediciton = model.predict(input_scaler)
    
    st.subheader('Cell Cluster Prediction')
    st.write('The cell cluster is :')

    if prediciton[0] ==0:
        st.write('<span class="diagnosis benign">Benign</span>', unsafe_allow_html=True )#Allow to parse HTML
    else:
        st.write('<span class="diagnosis malicious">Malicious</span>', unsafe_allow_html=True )#Allow to parse HTML

    
    st.write('Probability of being benign:', round(model.predict_proba(input_scaler)[0][0], 3))
    st.write('Probability of being malignant:', round(model.predict_proba(input_scaler)[0][1], 3))


def main():

    st.set_page_config(
       page_title = 'Breast Cancer Predector',
       page_icon = ':femal:',
       layout="wide",
       initial_sidebar_state='expanded',
      )
    with open('assets/style.css') as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html = True)
    
    input_data = add_sidebar()
    

    with st.container():
        st.title('Breast Concer Predector')
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")
        
    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_prediction(input_data)

   

if __name__ == '__main__':
    main()
