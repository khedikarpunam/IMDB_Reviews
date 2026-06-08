import streamlit as st
import joblib
import re
import nltk

# Download NLTK data (if not already downloaded in the environment where Streamlit runs)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punk_tab', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the trained model and vectorizer
try:
    log_reg_model = joblib.load('logistic_regression_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    st.success("Model and Vectorizer loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model or Vectorizer file not found. Please ensure 'logistic_regression_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory as your Streamlit app.")
    st.stop()

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define the text cleaning function (must be the same as used during training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove punctuation and numbers
    return text

def preprocess_text(text):
    cleaned_text = clean_text(text)
    tokens = nltk.word_tokenize(cleaned_text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Streamlit app layout
st.title('IMDB Movie Review Sentiment Predictor')
st.write('Enter a movie review below to get its sentiment prediction.')

user_input = st.text_area('Movie Review', '')

if st.button('Predict Sentiment'):
    if user_input:
        # Preprocess the input text
        processed_input = preprocess_text(user_input)

        # Transform the processed text using the loaded TF-IDF vectorizer
        input_vectorized = tfidf_vectorizer.transform([processed_input])

        # Make prediction
        prediction = log_reg_model.predict(input_vectorized)
        prediction_proba = log_reg_model.predict_proba(input_vectorized)

        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        positive_proba = prediction_proba[0][1] * 100
        negative_proba = prediction_proba[0][0] * 100

        st.write(f"### Predicted Sentiment: **{sentiment}**")
        st.write(f"Confidence: Positive - {positive_proba:.2f}% | Negative - {negative_proba:.2f}%")
    else:
        st.warning('Please enter a movie review to get a prediction.')
