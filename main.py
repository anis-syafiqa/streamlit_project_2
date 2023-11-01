import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer


# Load the model and the TF-IDF vectorizer
model_duration = joblib.load('duration_cluster_model.bin')
model_cpu = joblib.load('cpu_cluster_model.bin')
model_bytes = joblib.load('bytes_cluster_model.bin')
model_spilled = joblib.load('spilled_cluster_model.bin')
model_node = joblib.load('node_cluster_model.bin')

tfidf_vectorizer_duration = joblib.load('duration_tfidf_vectorizer.bin')
tfidf_vectorizer_cpu = joblib.load('cpu_tfidf_vectorizer.bin')
tfidf_vectorizer_bytes = joblib.load('bytes_tfidf_vectorizer.bin')
tfidf_vectorizer_spilled = joblib.load('spilled_tfidf_vectorizer.bin')
tfidf_vectorizer_node = joblib.load('node_tfidf_vectorizer.bin')

# Title of your app
st.markdown("<h1 style='text-align: center; color: GREEN;'>SQL Query Cost Predictor  📊 </h3>", unsafe_allow_html=True)

# Text input for the user
user_input = st.text_area('Enter your SQL query:')

def is_sql_query(text):
    # Define a more comprehensive SQL query pattern
    sql_pattern = r'\b(SELECT|INSERT\sINTO|UPDATE|DELETE|CREATE|ALTER|DROP|INSERT\sOVERWRITE|COMPUTE\sSTATS|INVALIDATE\sMETADATA|REFRESH|MERGE|EXEC)\b'

    # Check if the text matches the SQL query pattern
    if re.search(sql_pattern, text, re.IGNORECASE):
        return True

    return False

# Make predictions
if st.button('Predict'):
    # Check if the user input resembles a SQL query
    is_sql = is_sql_query(user_input)
    
    if is_sql:
        # Transform the user input      
        X_new_statement_duration = tfidf_vectorizer_duration.transform([user_input]).toarray()
        X_new_statement_cpu = tfidf_vectorizer_cpu.transform([user_input]).toarray()
        X_new_statement_bytes = tfidf_vectorizer_bytes.transform([user_input]).toarray()
        X_new_statement_node = tfidf_vectorizer_node.transform([user_input]).toarray()
        X_new_statement_spilled = tfidf_vectorizer_spilled.transform([user_input]).toarray()

        #Make predictions using the loaded models
        y_pred_duration = model_duration.predict(X_new_statement_duration).round().astype(int)  
        y_pred_cpu = model_cpu.predict(X_new_statement_cpu).round().astype(int)  
        y_pred_bytes = model_bytes.predict(X_new_statement_bytes).round().astype(int)  
        y_pred_node = model_node.predict(X_new_statement_node).round().astype(int)  
        y_pred_spilled = model_spilled.predict(X_new_statement_spilled).round().astype(int)  

        # Define the labels for your cluster names
        cluster_names_duration = {
            0: "Short Duration",
            1: "Long Duration",
            2: "Medium Duration"
        }
        cluster_names_cpu = {
            0: "Low CPU Usage",
            1: "High CPU Usage"
        }
        cluster_names_bytes = {
            0: "Low Data Transfer",
            1: "Very High Data Transfer",
            2: "Moderate Data Transfer",
            3: "High Data Transfer"
        }
        cluster_names_node = {
            0: "Moderate Memory Usage",
            1: "High Memory Usage",
            2: "Low Memory Usage"
        }
        cluster_names_spill = {
            0: "Low Memory Spill",
            1: "Very High Memory Spill",
            2: "High Memory Spill",
            3: "Moderate Memory Spill"
        }
        
        #Map cluster labels to meaningful names
        predicted_duration= cluster_names_duration.get(y_pred_duration[0], "Unknown")   
        predicted_cpu = cluster_names_cpu.get(y_pred_cpu[0], "Unknown")
        predicted_node = cluster_names_node.get(y_pred_node[0], "Unknown")
        predicted_spilled = cluster_names_spill.get(y_pred_spilled[0], "Unknown")
        predicted_bytes = cluster_names_bytes.get(y_pred_bytes[0], "Unknown")

        # Show the prediction
        st.markdown("<h1 style='text-align: center;color: GREEN;'>Prediction</h1>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        col1.markdown("### **Duration** ⏱️", unsafe_allow_html=True)
        col1.write(f'Prediction: {predicted_duration}')

        col2.markdown("### **CPU** 💻", unsafe_allow_html=True)
        col2.write(f'Prediction: {predicted_cpu}')

        col1, col2 = st.columns(2)

        col1.markdown("### **Memory Spilled** 💽", unsafe_allow_html=True)
        col1.write(f'Prediction: {predicted_spilled}')

        col2.markdown("### **Node Peak Memory** 📈", unsafe_allow_html=True)
        col2.write(f'Prediction: {predicted_node}')

        st.markdown("### **Bytes Streamed** 💾", unsafe_allow_html=True)        col1, col2 = st.columns(2)
        col1.write(f'Prediction: {predicted_bytes}')
        #col2.markdown('* **Bytes Streamed in range:** {}')


    else:
        st.write('Input is not a SQL query. Prediction is not performed.')

