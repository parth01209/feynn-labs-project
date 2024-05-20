import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load your dataset vectors from the .txt file
def load_dataset_vectors(file_path):
    with open(file_path, 'r') as file:
        dataset_vectors = [list(map(float, line.strip().split())) for line in file]
    return dataset_vectors

# Load your dataset vectors (replace 'your_dataset_vectors.txt' with the actual file path)
dataset_vectors = load_dataset_vectors('book_vectors.txt')

while True:
    # Get user input interactively
    user_input = input("You: ")

    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("Chatbot: Goodbye!")
        break

    # NLU: Tokenize and vectorize user input
    def vectorize_input(user_input):
        user_input_tokens = user_input.split()
        valid_tokens = [word for word in user_input_tokens if word in dataset_vectors]
        if not valid_tokens:
            return None  # Return None if no valid tokens found
        user_input_vector = np.mean([vector for vector in dataset_vectors if len(vector) == len(valid_tokens)], axis=0)
        return user_input_vector.reshape(1, -1)  # Reshape to 2D array

    user_input_vector = vectorize_input(user_input)

    if user_input_vector is None:
        print("Chatbot: I'm sorry, I couldn't understand your question.")
        continue

    # Information Retrieval: Find the most relevant answer directly from the dataset vectors
    def find_most_similar_answer(user_input_vector, dataset_vectors):
        similarity_scores = [cosine_similarity(user_input_vector, [vector])[0][0] for vector in dataset_vectors]
        most_similar_index = np.nanargmax(similarity_scores)
        return dataset_vectors[most_similar_index]  # Return the most similar vector

    most_similar_vector = find_most_similar_answer(user_input_vector, dataset_vectors)

    # You can convert the most similar vector back to a human-readable format, if needed
    # For example, if each line in 'your_dataset_vectors.txt' corresponds to a recommendation
    # Then you can directly print the most similar vector as the chatbot's response

    # Print the chatbot's response (as an example, printing the vector)
    print("Chatbot:", most_similar_vector)
