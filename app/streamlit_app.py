import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import plotly.graph_objects as go

#Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📝",
    layout="wide"
)

#Get paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#Load model artifacts
@st.cache_resource
def load_model_artifacts():
    model = joblib.load(PROJECT_ROOT / 'data/models/best_model.pkl')
    scaler = joblib.load(PROJECT_ROOT / 'data/models/scaler.pkl')
    features = joblib.load(PROJECT_ROOT / 'data/models/feature_names.pkl')
    return model, scaler, features

model, scaler, features = load_model_artifacts()

#Title
st.title("Customer Churn Prediction System")
st.markdown("""
Predict which customers are at risk of leaving using machine learning.
Input customer information and get instant churn risk score.
""")

#Tabs
tab1, tab2, tab3 = st.tabs(["Predict Churn Risk", "Model Performance", "About"])

#Tab 1: prediction
with tab1:
    st.header("Predict customer churn risk")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        partner = st.selectbox("Has partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    
    with col2:
        st.subheader("Services")
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet"])

    with col3:
        st.subheader("Account Info")
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
        total_charges = st.number_input("Total Charges ($)", 0, 10000, value=tenure*monthly_charges)
    
    #Predict button
    if st.button("Predict Churn Risk", type="primary", use_container_width=True):
        st.markdown("---")

        #prepare features matching the training data structure
        feature_dict = {
            'gender': 1 if gender == 'Male' else 0,
            'Partner': 1 if partner == 'Yes' else 0,
            'Dependents': 1 if dependents == 'Yes' else 0,
            'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
            'tenure': tenure,
            'PhoneService': 1 if phone_service == 'Yes' else 0,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'OnlineSecurity': 1 if online_security == 'Yes' else 0,
            'TechSupport': 1 if tech_support == 'Yes' else 0
        }

        #Add internet service one-hot
        feature_dict['InternetService_Fiber optic'] = 1 if internet_service == 'Fiber optic' else 0
        feature_dict['InternetService_No'] = 1 if internet_service == 'No' else 0

        #Add contract one-hot
        feature_dict['Contract_One year'] = 1 if contract == 'One year' else 0
        feature_dict['Contract_Two year'] = 1 if contract == 'Two year' else 0

        #Add other required features with defaults
        for feature in features:
            if feature not in feature_dict:
                feature_dict[feature] = 0
        
        #Create feature array
        feature_values = np.array([feature_dict[f] for f in features]).reshape(1, -1)
        feature_values_scaled = scaler.transform(feature_values)

        #Make prediction
        churn_prob = model.predict_proba(feature_values_scaled)[0][1]
        churn_pred = model.predict(feature_values_scaled)[0]

        #Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            if churn_pred == 1:
                st.metric("Churn risk", f"{churn_prob*100:.1f}%", "HIGH RISK")
            else:
                st.metric("Churn risk", f"{churn_prob*100:.1f}%", "LOW RISK")
        
        with col2:
            st.metric("Recommendation", "Immediate Action" if churn_prob > 0.5 else "Monitor")

        with col3:
            st.metric("Confidence", f"{max(churn_prob, 1-churn_prob)*100:.1f}%")

        #Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_prob*100,
            title={'text': 'Churn Risk Score'},
            delta={'reference': 50},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "darkred" if churn_prob > 0.5 else "green"},
                   'steps': [
                       {'range': [0, 25], 'color': "lightgreen"},
                       {'range': [25, 50], 'color': "yellow"},
                       {'range': [50, 75], 'color': "orange"},
                       {'range': [75, 100], 'color': "red"}],
                    'threshold': {'line': {'color': "red", 'width': 4},
                                  'thickness': 0.75,
                                  'value': 50}}))
        
        if churn_prob < 0.25:
            st.success("🟢 Low Risk Customer")

        elif churn_prob < 0.50:
            st.info("🔵 Moderate Risk Customer")

        elif churn_prob < 0.75:
            st.warning("🟠 High Risk Customer")

        else:
            st.error("🔴 Critical Risk Customer")

        st.progress(float(churn_prob))

        st.subheader("Risk Factors")

        if tenure < 12:
            st.write("• New customer")

        if contract == "Month-to-month":
            st.write("• Month-to-month contract")

        if tech_support == "No":
            st.write("• No technical support")

        if online_security == "No":
            st.write("• No online security")
            
        st.plotly_chart(fig, use_container_width=True)
    
#Tab 2: Model performance
with tab2:
    st.header("Model Performance")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", "80%")
    with col2:
        st.metric("Precision", "66%")
    with col3:
        st.metric("ROC-AUC", "0.845")
    
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        try:
            st.image(str(PROJECT_ROOT / 'data/confusion_matrix.png'))
        except:
            st.info("Run evaluation notebook to generate plots")
        
    with col2:
        try:
            st.image(str(PROJECT_ROOT / 'data/roc_curve.png'))
        except:
            st.info("Run evaluation notebook to generate plots")
        
    st.markdown("---")

    try:
        st.image(str(PROJECT_ROOT / 'data/logistic_feature_importance.png'))
    except:
        st.info("Run evaluation notebook to generate plots")

#Tab 3: About
with tab3:
    st.header("About this project")

    st.markdown("""
    ## Customer Churn Prediction System

    ### What is Churn?
    Churn occurs when a Customer stops using a service. For telecom companies,
    predicting churn helps identify at-risk customers for proactive retention.
    
    ### How it works
                
    1. Data Collection - Historical customer data including:
    - Demographics (age, gender, family status)
    - Services (internet, phone, streaming)
    - Billing (monthly charges, contract type)
    - Churn outcome (left or stayed)
    
    2. Feature Engineering - Created predictive features:
    - Customer characteristics (demographics)
    - Services adoption patterns
    - Billing indicators (high charges = high risk)
    - Tenure groups (new customers more likely to churn)
    
    3. Model Training - Tested multiple algorithms:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    
    Selected best based on ROC-AUC (balances false positives/negatives)
    
    4. Prediction - Input customer info → Get churn risk score:
    
    ### Key Findings
    
    - **Tenure** - Strongest predictor (new customers at high risk)
    - **Contract Type** - Month-to-month contracts have highest churn
    - **Internet Type** - Fiber optic customers churn more
    - **Tech Support** - Customers without support churn more
    
    ### Model Performance
    
    - **Accuracy**: 80.5% (overall correctness)
    - **Precision**: 66.7% (avoid false retention efforts)
    - **Recall**: 52.9% (catch actual churners)
    - **ROC-AUC**: 0.845 (excellent ranking ability)
    
    ### Use Cases
    
    - **Retention campaigns** - Target high-risk customers
    - **Pricing decisions** - Adjust for high-churn segments
    - **Product improvements** - Understand pain points
    - **Customer segmentation** - Different strategies per segment
    
    ### Technologies
    
    - Python, pandas, scikit-learn
    - Streamlit for deployment
    - Kaggle dataset
    
    ---
    
    ### Author
    
    **Eduardo Arriaga Alejandre**
    
    Telematics Engineer building data science and ML engineering skills.
    
    - [GitHub](https://github.com/PraetorClaudius)
    - [LinkedIn](https://www.linkedin.com/in/eduardo-arriaga-230156295/)
    
    ### Project Repository
    
    Full source code, notebooks, and documentation available on GitHub:
    https://github.com/PraetorClaudius/customer-churn-prediction
    
    ---
    
    Built as portfolio project demonstrating classification ML workflow.
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p> Customer Churn Predictor | ML Classification System</p>
</div>
""", unsafe_allow_html=True)