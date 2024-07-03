import random
import json
import pickle
import nltk
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents and model data
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Assuming 'words' should match the number of features expected by your model
input_dim = len(words)

# Modify the dummy_x initialization based on the expected input shape
dummy_x = np.zeros((1, input_dim))

# Ensure your model's architecture is correctly set up
# Note: Remove the input_shape argument from the first Dense layer
model = Sequential([
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(classes), activation='softmax')  # Adjust output_dim to match len(classes)
])

# Compile the model with appropriate optimizer and loss function
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Now, you can evaluate the model with the correct input shape and dummy_y
dummy_y = np.array([[0, 0, 0, 1]])  # Example: one-hot encoded target for a single instance

model.evaluate(dummy_x, dummy_y, verbose=0)

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

    
    ERROR_THRESHOLD = 0.15  # Adjusted threshold
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    

    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list



def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

if __name__ == "__main__":
    print("Chatbot is ready! Type 'exit' to end the chat.")
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Exiting chat. Goodbye!")
                break
            intents_list = predict_class(user_input)
            if intents_list:
                response = get_response(intents_list, intents)
                print(f"Bot: {response}")
            else:
                print("Bot: I'm not sure how to respond to that.")
    except (KeyboardInterrupt, EOFError, SystemExit):
        print("\nExiting chat. Goodbye!")
