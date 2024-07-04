# app.py
import streamlit as st
import numpy as np
import pickle
import os
import random
import json
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from pdfextract import extract_text_from_pdf


nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()


model = load_model('chatbot_model_with_pdf.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


intents = json.loads(open('intents.json').read())

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.15
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list,  intents_json ):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def main():
    st.title("MasterPDF")
    st.write("Welcome to MasterPDF. Ask me anything based on the uploaded PDF content.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        cleaned_text = extract_text_from_pdf("uploaded_file.pdf")
        st.write("PDF content processed. You can now ask questions based on the PDF content.")
        with open('cleaned_text.txt', 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    user_input = st.text_input("You:", "")

    if user_input:
        st.session_state['chat_history'].append(f"You: {user_input}")
        intents_list = predict_class(user_input)
        if intents_list:
            response = get_response(intents_list, intents)
            st.session_state['chat_history'].append(f"Bot: {response}")
        else:
            st.session_state['chat_history'].append("Bot: I'm not sure how to respond to that.")

    if st.session_state['chat_history']:
        for chat in st.session_state['chat_history']:
            st.write(chat)

if __name__ == "__main__":
    main()
