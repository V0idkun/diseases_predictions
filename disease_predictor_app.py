import os
from datetime import datetime

import joblib
import pandas as pd
import tensorflow as tf
import streamlit as st
import numpy as np

st.set_page_config(
    page_title='Disease Predictor',
    page_icon='🧠',
    layout='wide',
)

BASE_DIR = os.path.dirname(__file__)
LOG_FILE = os.path.join(BASE_DIR, 'user_predictions_log.csv')
ADMIN_KEY = os.environ.get('DISEASE_APP_ADMIN_KEY', 'admin123')


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(os.path.join(BASE_DIR, 'disease_predictor_model.keras'))


@st.cache_resource
def load_label_encoder():
    return joblib.load(os.path.join(BASE_DIR, 'label_encoder.pkl'))


@st.cache_resource
def load_metadata():
    return joblib.load(os.path.join(BASE_DIR, 'disease_metadata.joblib'))


@st.cache_data
def load_uploaded_data(uploaded_file):
    return pd.read_csv(uploaded_file)


def ensure_log_file():
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(
            columns=[
                'timestamp',
                'user_id',
                'input_type',
                'selected_symptoms',
                'symptom_vector',
                'sample_count',
                'predicted_disease',
                'confidence',
                'upload_filename',
                'note',
            ]
        )
        df.to_csv(LOG_FILE, index=False)


def log_prediction(entry):
    ensure_log_file()
    pd.DataFrame([entry]).to_csv(LOG_FILE, mode='a', header=False, index=False)


def selected_symptoms_list(user_inputs, features):
    return [name for name, selected in zip(features, user_inputs) if selected]


model = load_model()
le = load_label_encoder()
metadata = load_metadata()
feature_names = metadata['feature_name']

st.title('DISEASE PREDICTION FROM SYMPTOMS 🦠💊')

with st.expander('ℹ️ Model Info'):
    st.write('Model trained on 41 disease classes with 132 symptoms.')
    st.write('Accuracy: ~99% on validation data.')
    st.write('Use the tabs to choose a single prediction or upload a dataset for batch scoring.')

main_tab, upload_tab, admin_tab = st.tabs(['Symptoms Inputs', 'Data Upload', 'Admin Dashboard'])

with main_tab:
    st.subheader('Select your symptoms to get your predicted disease:')
    user_id = st.text_input('Your name or ID (optional)', placeholder='Name, email, or participant ID')

    user_inputs = []
    num_cols = 4
    cols = st.columns(num_cols)
    for i, feature in enumerate(feature_names):
        with cols[i % num_cols]:
            label = feature.replace('_', ' ').title()
            checked = st.checkbox(label, key=f'{feature}_{i}')
        user_inputs.append(1 if checked else 0)

    selected = selected_symptoms_list(user_inputs, feature_names)
    st.write(f'Selected symptoms: **{len(selected)}**')

    if st.button('Predict Disease'):
        if len(selected) == 0:
            st.warning('Please select at least one symptom before predicting.')
        else:
            prediction = model.predict(np.array(user_inputs).reshape(1, -1))
            predicted_class = np.argmax(prediction)
            disease_name = le.inverse_transform([predicted_class])[0]
            confidence = float(np.max(prediction) * 100)
            st.success(f'Predicted Disease: **{disease_name}**')
            st.info(f'Confidence: {confidence:.2f}%')

            if confidence < 50:
                st.markdown(':red[⚠️ Low Confidence - The model is not very confident in this prediction.]')
            elif confidence < 60:
                st.markdown(':orange[🟠 Moderate Confidence - The model is somewhat confident in this prediction.]')
            elif confidence < 80:
                st.markdown(':blue[✅ Confidence - The model is reasonably confident in this prediction.]')
            elif confidence < 95:
                st.markdown(':green[✅✅ High Confidence - The model is highly confident in this prediction.]')
            else:
                st.markdown(':green[💯 Very High Confidence - The model is almost certain in this prediction!]')

            log_prediction({
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': user_id.strip() or 'anonymous',
                'input_type': 'single',
                'selected_symptoms': ';'.join(selected),
                'symptom_vector': ','.join(str(val) for val in user_inputs),
                'sample_count': 1,
                'predicted_disease': disease_name,
                'confidence': confidence,
                'upload_filename': '',
                'note': 'single symptom prediction',
            })
            st.success('Your prediction has been recorded for review.')

with upload_tab:
    st.subheader('Batch prediction via CSV upload')
    upload_user_id = st.text_input('Your name or ID (optional)', key='upload_user_id', placeholder='Name, email, or participant ID')
    st.write('Upload a CSV file with symptom columns matching the model input. Missing symptom columns will be filled with zeros.')

    template_df = pd.DataFrame([{feature: 0 for feature in feature_names}])
    csv_template = template_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download upload template CSV', csv_template, 'symptom_upload_template.csv')

    uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])

    if uploaded_file is not None:
        try:
            df = load_uploaded_data(uploaded_file)
            missing = [col for col in feature_names if col not in df.columns]
            if missing:
                st.warning('Missing columns were added with zeros.')
                for col in missing:
                    df[col] = 0

            df = df[feature_names]
            st.success('The file was uploaded successfully')
            st.subheader('Dataset Preview')
            st.write(df.head())
            st.write(f'Rows to score: **{len(df)}**')

            if st.button('Predict Uploaded Data'):
                with st.spinner('🧠 Scoring uploaded dataset...'):
                    if list(df.columns) != list(feature_names):
                        st.error('🚫 The uploaded file does not match the required feature format.')
                    else:
                        prediction = model.predict(df)
                        predicted_classes = np.argmax(prediction, axis=1)
                        disease_names = le.inverse_transform(predicted_classes)
                        df['Predicted_Diseases'] = disease_names
                        st.subheader('Prediction Results')
                        st.dataframe(df)

                        summary = pd.Series(disease_names).value_counts().to_dict()
                        log_prediction({
                            'timestamp': datetime.utcnow().isoformat(),
                            'user_id': upload_user_id.strip() or 'anonymous',
                            'input_type': 'upload',
                            'selected_symptoms': '',
                            'symptom_vector': '',
                            'sample_count': int(len(df)),
                            'predicted_disease': 'multiple',
                            'confidence': float(np.max(prediction) * 100),
                            'upload_filename': uploaded_file.name,
                            'note': f'upload summary: {summary}',
                        })
                        csv_result = df.to_csv(index=False).encode('utf-8')
                        st.download_button('Download results CSV', csv_result, 'predictions.csv')
                        st.success('Upload prediction results have been recorded.')
        except Exception as e:
            st.error(f'There was an error processing the file: {e}')

with admin_tab:
    st.subheader('Admin Dashboard')
    st.markdown('Enter the admin key to view submission history and download logs.')
    admin_input = st.text_input('Admin Access Key', type='password')

    if admin_input:
        if admin_input == ADMIN_KEY:
            ensure_log_file()
            log_df = pd.read_csv(LOG_FILE)
            log_df['timestamp'] = pd.to_datetime(log_df['timestamp'], errors='coerce')

            st.write('### Filters')
            filter_types = st.multiselect('Input type', ['single', 'upload'], default=['single', 'upload'])
            user_search = st.text_input('Search user ID')
            min_date = st.date_input('Start date', value=log_df['timestamp'].min().date() if not log_df.empty else datetime.utcnow().date())
            max_date = st.date_input('End date', value=log_df['timestamp'].max().date() if not log_df.empty else datetime.utcnow().date())

            filtered = log_df[log_df['input_type'].isin(filter_types)]
            if not filtered.empty:
                filtered = filtered[filtered['timestamp'].dt.date.between(min_date, max_date)]
            if user_search:
                filtered = filtered[filtered['user_id'].str.contains(user_search, case=False, na=False)]

            st.write(f'### Showing {len(filtered)} submissions')
            st.dataframe(filtered.sort_values('timestamp', ascending=False).reset_index(drop=True))

            if not filtered.empty:
                st.write('### Prediction counts')
                st.bar_chart(filtered['predicted_disease'].value_counts())

            with open(LOG_FILE, 'rb') as f:
                st.download_button('Download full log CSV', f, 'user_predictions_log.csv')
        else:
            st.error('Invalid admin key.')
