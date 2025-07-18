# Streamlit Resume Category Classifier App

import streamlit as st
import pickle
import re
import PyPDF2
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load models
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

category_map = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

def clean(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
    clean_text = url_pattern.sub('', text)
    clean_text = email_pattern.sub('', clean_text)
    clean_text = re.sub('[^\w\s]', '', clean_text)
    clean_text = ' '.join(word for word in clean_text.split() if word.lower() not in stop_words)
    return clean_text

def extract_text(uploaded_file):
    if uploaded_file.name.lower().endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        return uploaded_file.read().decode('utf-8', errors='ignore')

st.title("Resume Category Classifier")
st.write("Upload your resume (txt or pdf) to predict its category.")

uploaded_file = st.file_uploader("Choose a resume file", type=["txt", "pdf"])

if uploaded_file is not None:
    try:
        text = extract_text(uploaded_file)
        cleaned = clean(text)
        features = tfidf.transform([cleaned])
        pred_id = model.predict(features)[0]
        category = category_map.get(pred_id, "Unknown")
        st.success(f"Predicted Category: {category}")
    except Exception as e:
        st.error(f"Error: {e}")