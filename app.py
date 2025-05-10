import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

model = pickle.load(open('xgb_mddel.pkl', 'rb'))

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title('🔍 Credit Card Fraud Detection App')
st.markdown("Upload a CSV file to detect fraudulent credit card transactions using a trained XGBoost model.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Sample")
        st.dataframe(data.head())

        expected_features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        missing_cols = [col for col in expected_features if col not in data.columns]

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            if st.button("Run Fraud Detection"):
                X = data[expected_features]
                predictions = model.predict(X)
                data['Prediction'] = predictions
                fraud_count = sum(predictions)

                st.success(f"Detection complete. {fraud_count} fraudulent transactions found.")
                st.write(data['Prediction'].value_counts().rename_axis('Class').reset_index(name='Count'))

                st.subheader("Fraudulent Transactions")
                st.dataframe(data[data['Prediction'] == 1].head(10))

                csv_download = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Result as CSV",
                    data=csv_download,
                    file_name='fraud_detection_result.csv',
                    mime='text/csv'
                )

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
else:
    st.info("Please upload a CSV file to proceed.")
