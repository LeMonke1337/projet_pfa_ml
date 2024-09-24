import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")

df = pd.read_csv("resumme_v3.csv")

df["combined_text"] = (
    df["education"].fillna("")
    + " "
    + df["experience"].fillna("")
    + " "
    + df["skills"].fillna("")
)


def preprocess(text):
    # Lowercase the text
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    # Tokenize the text into words
    tokens = word_tokenize(text)
    return tokens


df["tokens"] = df["combined_text"].apply(preprocess)


def load_glove_vectors(glove_file):
    glove_dict = {}
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype="float32")
            glove_dict[word] = vector
    return glove_dict


glove_file = "glove.6B.300d.txt"
glove_vectors = load_glove_vectors(glove_file)


def get_average_glove(tokens, glove_vectors, vector_size):
    word_vectors = [glove_vectors[word] for word in tokens if word in glove_vectors]

    if len(word_vectors) == 0:
        return np.zeros(vector_size)

    return np.mean(word_vectors, axis=0)


vector_size = 300
df["glove_vector"] = df["tokens"].apply(
    lambda tokens: get_average_glove(tokens, glove_vectors, vector_size)
)

glove_matrix = np.stack(df["glove_vector"].values)


def recommend_category_with_glove(custom_profile, df, glove_vectors, vector_size):
    custom_profile_text = f"education : {custom_profile['education']} experience : {custom_profile['experience']} skills : {custom_profile['skills']}"
    custom_profile_tokens = preprocess(custom_profile_text)
    custom_profile_vector = get_average_glove(
        custom_profile_tokens, glove_vectors, vector_size
    )

    similarity_scores = cosine_similarity(
        [custom_profile_vector], glove_matrix
    ).flatten()

    df["similarity_score"] = similarity_scores

    category_similarity = (
        df.groupby("category")["similarity_score"].mean().reset_index()
    )

    top_category = category_similarity.sort_values(
        by="similarity_score", ascending=False
    ).iloc[0]

    return top_category["category"], top_category["similarity_score"]


custom_tech_profile = {
    "education": "Bachelor of Science in Computer Science",
    "experience": "4 years of experience as a software engineer, focusing on web development, cloud computing, and microservices architecture. Led a team in developing scalable web applications and integrated cloud-based solutions.",
    "skills": "Python, Java, JavaScript, AWS, Docker, Kubernetes, CI/CD, REST APIs, Agile methodologies",
}

recommended_category, similarity_score = recommend_category_with_glove(
    custom_tech_profile, df, glove_vectors, vector_size
)

print(
    f"Recommended Category: {recommended_category} (Similarity Score: {similarity_score:.2f})"
)
