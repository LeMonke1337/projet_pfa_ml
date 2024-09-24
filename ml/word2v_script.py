import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.tokenize import word_tokenize
import numpy as np
import nltk

nltk.download("punkt")

# Load your dataset
df = pd.read_csv("resumme_v3.csv")

# Combine 'education', 'experience', and 'skills' into a single text field
df["combined_text"] = (
    df["education"].fillna("")
    + " "
    + df["experience"].fillna("")
    + " "
    + df["skills"].fillna("")
)


# Preprocess the text: lowercasing and tokenization
def preprocess(text):
    # Lowercase the text
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    # Tokenize the text into words
    tokens = word_tokenize(text)
    return tokens


# Apply preprocessing to the 'combined_text' column
df["tokens"] = df["combined_text"].apply(preprocess)

# Prepare the list of tokenized sentences for Word2Vec
tokenized_sentences = df["tokens"].tolist()

# Train a Word2Vec model
word2vec_model = Word2Vec(
    sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4
)


# Function to get the average Word2Vec vector for a document
def get_average_word2vec(tokens, model, vector_size):
    # Get the word vectors for the tokens that exist in the model's vocabulary
    word_vectors = [model.wv[word] for word in tokens if word in model.wv]

    if len(word_vectors) == 0:
        return np.zeros(
            vector_size
        )  # Return a zero vector if none of the words are in the vocabulary

    # Return the average of the word vectors
    return np.mean(word_vectors, axis=0)


# Apply the function to all resumes in the dataset
df["word2vec_vector"] = df["tokens"].apply(
    lambda tokens: get_average_word2vec(tokens, word2vec_model, 100)
)


# Function to recommend the closest category using Word2Vec and Cosine Similarity
def recommend_category_with_word2vec(custom_profile, df):
    # Preprocess the custom profile and get the Word2Vec vector
    custom_profile_text = f"education : {custom_profile['education']} experience : {custom_profile['experience']} skills : {custom_profile['skills']}"
    custom_profile_tokens = preprocess(custom_profile_text)
    custom_profile_vector = get_average_word2vec(
        custom_profile_tokens, word2vec_model, 100
    )

    # Calculate cosine similarity between the custom profile vector and all the Word2Vec vectors in the dataset
    similarity_scores = cosine_similarity(
        [custom_profile_vector], list(df["word2vec_vector"].values)
    ).flatten()

    # Add the similarity scores to the dataframe
    df["similarity_score"] = similarity_scores

    # Group by category and calculate the average similarity for each category
    category_similarity = (
        df.groupby("category")["similarity_score"].mean().reset_index()
    )

    # Sort by similarity and get the most similar category
    top_category = category_similarity.sort_values(
        by="similarity_score", ascending=False
    ).iloc[0]

    return top_category["category"], top_category["similarity_score"]


# Custom profile data (technology major example)
custom_tech_profile = {
    "education": "Bachelor of Science in Computer Science",
    "experience": "4 years of experience as a software engineer, focusing on web development, cloud computing, and microservices architecture. Led a team in developing scalable web applications and integrated cloud-based solutions.",
    "skills": "Python, Java, JavaScript, AWS, Docker, Kubernetes, CI/CD, REST APIs, Agile methodologies",
}

# Get the recommended category based on the custom profile using Word2Vec and cosine similarity
recommended_category, similarity_score = recommend_category_with_word2vec(
    custom_tech_profile, df
)

# Display the result
print(
    f"Recommended Category: {recommended_category} (Similarity Score: {similarity_score:.2f})"
)
