import joblib
import pandas as pd
import tensorflow as tf
import streamlit as st
import numpy as np


@st.cache_resource
def load_model():
    return tf.keras.models.load_model('disease_predictor_model.keras')

@st.cache_resource
def load_label_encoder():
    return joblib.load('label_encoder.pkl')
@st.cache_resource
def load_metadata():
    return joblib.load('disease_metadata.joblib')

model = load_model()
le = load_label_encoder()
metadata = load_metadata()
feature_names = metadata['feature_name']

st.title('DISEASE PREDICTION FROM SYMPTOMS 🦠💊')

tab,tab1 = st.tabs(['Symptoms Inputs','Data Upload'])

with st.expander("ℹ️ Model Info"):
    st.write("Model trained on 41 disease classes with 132 symptoms.")
    st.write("Accuracy: ~99% on validation data.")

with tab:
    st.subheader('Select your symptoms to get your predicted disease:')
    user_inputs = []
    num_cols = 4
    cols = st.columns(num_cols)
    for i, feature in enumerate(feature_names): 
        with cols[i % num_cols]:
            label = feature.replace('_', ' ').title()
            checked = st.checkbox(label, key=f"{feature}_{i}")
        user_inputs.append(1 if checked else 0)

    input_array = np.array(user_inputs).reshape(1,-1)

    if st.button('Predict Disease'):
        if all(pred == 0 for pred in user_inputs):
            st.warning('Please select at least one symptom before predicting.')
        else:
            prediction = model.predict(input_array)
            predicted_classes = np.argmax(prediction)
            disease_name = le.inverse_transform([predicted_classes])
            confidence = np.max(prediction)*100
            st.success(f'Predicted Disease: **{disease_name}**')
            st.info(f'Confidence: {confidence:.2f}%')

            if confidence < 50:
                st.markdown(":red[⚠️ Low Confidence - The model is not very confident in this prediction.]")
            elif 50 <= confidence < 60:
                st.markdown(":orange[🟠 Moderate Confidence - The model is somewhat confident in this prediction.]")
            elif 60 <= confidence < 80:
                st.markdown(":blue[✅ Confidence - The model is reasonably confident in this prediction.]")
            elif 80 <= confidence < 95:
                st.markdown(":green[✅✅ High Confidence - The model is highly confident in this prediction.]")
            else:
                st.markdown(":green[💯 Very High Confidence - The model is almost certain in this prediction!]")

with tab1: 

    uploaded_file = st.file_uploader('Upload a csv file', type=['csv'])

    @st.cache_data
    def load_uploaded_data(uploaded_file):
            return pd.read_csv(uploaded_file)
    
    if uploaded_file is not None:
        try:
            df = load_uploaded_data(uploaded_file)
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[feature_names]
            st.success('The file was uploaded successfully')
            st.subheader('Dataset Preview')
            st.write(df.head())
            if st.button('Predict'):
                with st.spinner('🧠 Model is predicting...'):
                    if list(df.columns) != feature_names:
                        st.error("🚫 The uploaded file doesn't match the required feature format.")
                    else:
                        prediction = model.predict(df)
                        predicted_classes = np.argmax(prediction,axis=1)
                        disease_name = le.inverse_transform(predicted_classes)
                        df['Predicted_Diseases'] = disease_name
                        st.subheader('Predictions From Dataset Given :')
                        st.dataframe(df)
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button('Download Results as CSV', csv, 'predictions.csv')
        except Exception as e:
            st.error(f'They was an error processing the file {e}')