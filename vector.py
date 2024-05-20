import gensim
from gensim.models import Word2Vec
import numpy as np

# Sample text data
text_data = [
    "Ayurveda is a traditional system of medicine.",
    "Ayurvedic herbs have many health benefits.",
    "Yoga and meditation are integral parts of Ayurveda.",
    "Doshas, like Vata, Pitta, and Kapha, play a vital role in Ayurvedic medicine.",
]

# Tokenize the text (convert sentences to lists of words)
tokenized_data = [sentence.split() for sentence in text_data]

# Define Word2Vec model parameters
vector_size = 10  # Size of the word vectors
window_size = 5   # Maximum distance between the current and predicted word within a sentence
min_count = 1     # Ignores words with a lower frequency than this

model = Word2Vec(tokenized_data, vector_size=vector_size, window=window_size, min_count=min_count)

# Function to convert a sentence to a vector
def sentence_to_vector(sentence, model):
    words = sentence.split()
    vector = np.zeros(model.vector_size)  # Initialize an empty vector
    for word in words:
        if word in model.wv:
            vector += model.wv[word]
    return vector

# Convert a sentence to a vector
input_sentence = "Ayurvedic herbs have many health benefits."
vectorized_sentence = sentence_to_vector(input_sentence, model)

print("Input Sentence:", input_sentence)
print("Vector Representation:", vectorized_sentence)

