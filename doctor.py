import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
import numpy as np

# Sample dataset (for demonstration purposes)
input_text = ["What are some Ayurvedic remedies for a headache?",
              "Tell me about the Vata-Pitta-Kapha balance.",
              "How can I improve my digestion?"]

output_text = ["For a headache, you can try applying a paste of sandalwood on your forehead. It has cooling properties.",
               "The Vata-Pitta-Kapha balance is essential for maintaining health. Vata is associated with movement, Pitta with digestion and metabolism, and Kapha with stability.",
               "To improve digestion, consider drinking ginger tea before meals and avoiding heavy, greasy foods."]

# Tokenization
input_tokens = [sentence.split() for sentence in input_text]
output_tokens = [sentence.split() for sentence in output_text]

# Create an empty vocabulary set
vocab_set = set()

# Build vocabulary from both input and output tokens
for tokens in input_tokens + output_tokens:
    vocab_set.update(tokens)

# Convert the set to a list and sort it to maintain consistent word indices
vocab_list = sorted(list(vocab_set))

# Train Word2Vec model
model = Word2Vec(sentences=input_tokens + output_tokens, vector_size=100, window=5, min_count=1, sg=0)

# Function to vectorize a sentence
def vectorize_sentence(sentence, model):
    vectors = []
    for word in sentence:
        if word in model.wv:
            vectors.append(model.wv[word])
    return np.mean(vectors, axis=0)

# Vectorize input and output sequences
input_seq = [vectorize_sentence(sentence, model) for sentence in input_tokens]
output_seq = [vectorize_sentence(sentence, model) for sentence in output_tokens]

# Pad sequences for equal length (if needed)
input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq)
output_seq = tf.keras.preprocessing.sequence.pad_sequences(output_seq)

# Pad sequences for equal length
input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq)
output_seq = tf.keras.preprocessing.sequence.pad_sequences(output_seq)

# Define the sequence-to-sequence model
input_layer = Input(shape=(input_seq.shape[1],))
encoder = LSTM(128)(input_layer)
decoder = Dense(len(output_vocab), activation='softmax')(encoder)

model = Model(inputs=input_layer, outputs=decoder)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model (for demonstration purposes, use a small dataset)
model.fit(input_seq, output_seq, epochs=100)

# Generate a response given a user query
user_query = "What can I do for better sleep?"
user_query_seq = [input_vocab.index(word) for word in user_query.split()]
user_query_seq = tf.keras.preprocessing.sequence.pad_sequences([user_query_seq], maxlen=input_seq.shape[1])

response_seq = model.predict(user_query_seq)
response_text = ' '.join([output_vocab[np.argmax(token)] for token in response_seq[0]])

# Print the response in the "doctor" persona
print(f"Doctor Persona: {doctor_persona}")
print(f"User Query: {user_query}")
print(f"Response: {response_text}")
