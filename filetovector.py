#print 300 vectors for all respectively(word,sentence,txt file)

import numpy as np
import nltk
import string
from gensim.models import Word2Vec

# Download NLTK data (if not already downloaded)
nltk.download('punkt')

# Step 1: Read and preprocess the text data
with open('data.txt', 'r', encoding='utf-8') as file:
    text_data = file.read()

# Preprocess the text (tokenization, lowercase, punctuation removal, etc.) using NLTK
from nltk.tokenize import word_tokenize, sent_tokenize
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sent_tokenize(text_data)]

# Remove punctuation from tokens
table = str.maketrans('', '', string.punctuation)
tokenized_text = [[word.translate(table) for word in sentence] for sentence in tokenized_sentences]

# Step 2: Train a Word2Vec model
model = Word2Vec(tokenized_text, vector_size=300, window=5, min_count=1)

# Optionally, save the model for future use
model.save('word2vec_model')

# Step 3: Use the model to obtain vector representations
# For example, to get the vector for a specific word:
word_to_lookup = 'ayurveda'  # Change this to the word you want to look up
vector_for_word = model.wv[word_to_lookup]  # Use lowercase
print(f"Vector for the word '{word_to_lookup}':\n{vector_for_word}")

# To get the vector for a list of words (e.g., a sentence):
sentence_to_convert = "Ayurveda therapies have varied and evolved over more than two millennia."  # Change this to the sentence you want to convert
# Tokenize, lowercase, and remove punctuation
sentence_tokens = [word.translate(table) for word in word_tokenize(sentence_to_convert.lower())]
vector_for_sentence = [model.wv[word] for word in sentence_tokens if word in model.wv]
print(f"Vector for the sentence:\n{vector_for_sentence}")

# To obtain the vector for the entire document by averaging the vectors of its words:
def average_word_vectors(words, model):
    feature_vector = np.zeros((model.vector_size,), dtype="float32")
    nwords = 0
    for sentence in words:
        for word in sentence:
            if word in model.wv:
                nwords += 1
                feature_vector = np.add(feature_vector, model.wv[word])
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
    return feature_vector

document_vector = average_word_vectors(tokenized_text, model)
print(f"Vector for the entire document:\n{document_vector}")
