import random
import json
import pickle
import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load intents JSON
intents = json.loads(open('intents.json').read())

# Load extracted text
with open('cleaned_text.txt', 'r', encoding='utf-8') as f:
    extracted_text = f.read()


# Initialize data structures
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Process existing intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Tokenize and preprocess extracted text
pdf_words = nltk.word_tokenize(extracted_text)
pdf_words = [lemmatizer.lemmatize(word.lower()) for word in pdf_words if word not in ignore_letters]

# Assume all extracted text belongs to a new intent called "pdf_info"
pdf_documents = [(pdf_words, 'pdf_info')]

# Add new words and documents to existing data
words.extend(pdf_words)
documents.extend(pdf_documents)
if 'pdf_info' not in classes:
    classes.append('pdf_info')

# Remove duplicates and sort
words = sorted(set(words))
classes = sorted(set(classes))

# Save updated words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Build model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save model
model.save('chatbot_model_with_pdf.h5', hist)
print('done')
