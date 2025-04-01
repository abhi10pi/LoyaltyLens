import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load encoder and scaler
with open('onehot_encoder_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit App Start
st.set_page_config(page_title='LoyaltyLens', layout='centered', page_icon='💳')
st.title('💳 LoyaltyLens - Customer Churn Predictor')
st.markdown("""
    **Predict customer churn probability based on demographic and financial data.**
    
    Adjust the parameters and click **Check Churn** to see the result.
""")

# User input UI
with st.form(key='churn_form'):
    col1, col2 = st.columns(2)
    
    with col1:
        geography = st.selectbox('🌍 Geography', label_encoder_geo.categories_[0])
        gender = st.selectbox('⚤ Gender', label_encoder_gender.classes_)
        age = st.slider('🎂 Age', 18, 92, 30)
        balance = st.number_input('💰 Balance', min_value=0.0, value=50000.0, step=1000.0)
        credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=900, value=650)
    
    with col2:
        estimated_salary = st.number_input('💵 Estimated Salary', min_value=0.0, value=60000.0, step=1000.0)
        tenure = st.slider('📅 Tenure (years)', 0, 10, 5)
        num_of_products = st.slider('📦 Number of Products', 1, 4, 2)
        has_cr_card = st.selectbox('💳 Has Credit Card?', [0, 1])
        is_active_member = st.selectbox('🔥 Is Active Member?', [0, 1])
    
    check_button = st.form_submit_button(label='🔍 Check Churn')

if check_button:
    # Prepare input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encoding for 'Geography'
    geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

    # Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scaling the input data
    input_scaled = scaler.transform(input_data)

    # Predict the Churn
    prediction = model.predict(input_scaled)
    prediction_proba = prediction[0][0]

    st.markdown("---")
    st.subheader("📊 Prediction Result")
    st.metric(label='Churn Probability', value=f'{prediction_proba:.2%}')
    
    if prediction_proba > 0.5:
        st.error('🚨 The Customer is **likely** to churn!')
    else:
        st.success('✅ The Customer is **not likely** to churn.')