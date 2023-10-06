import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Initialize Hugging Face model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Streamlit app
st.title("Sentiment Analysis with Hugging Face")

user_input = st.text_area("Enter text here")

if user_input:
    # Tokenize the user input and run through the model
    tokens = tokenizer(user_input, padding=True, truncation=True, return_tensors="pt")
    output = model(**tokens)
    
    # Determine the sentiment
    _, predicted_class = output.logits[0].max(0)
    
    if predicted_class == 0:
        st.write("The sentiment is negative.")
    elif predicted_class == 1:
        st.write("The sentiment is neutral.")
    else:
        st.write("The sentiment is positive.")
