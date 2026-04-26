import streamlit as st
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize
from textblob import TextBlob

# Download tokenizer
nltk.download('punkt')
nltk.download('punkt_tab')

st.title("Grammar Error Detection Tool")
st.write("Enhance Academic and Professional Communication")

# Load grammar correction model
model_name = "vennify/t5-base-grammar-correction"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# User input
text = st.text_area("Enter a sentence or paragraph")

if st.button("Check Grammar"):

    if text.strip() == "":
        st.warning("Please enter text")

    else:
        corrected_spelling = str(TextBlob(text).correct())
        sentences = sent_tokenize(text)

        corrected_sentences = []

        for sentence in sentences:

            prompt = "grammar: " + sentence

            inputs = tokenizer(prompt, return_tensors="pt", padding=True)

            outputs = model.generate(**inputs, max_length=128)

            corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)

            corrected_sentences.append(corrected)

        corrected_paragraph = " ".join(corrected_sentences)

        st.subheader("Corrected Text")

        st.success(corrected_paragraph)