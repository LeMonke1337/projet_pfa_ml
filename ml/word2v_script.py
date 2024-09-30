import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.tokenize import word_tokenize
import numpy as np
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
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    return tokens


df["tokens"] = df["combined_text"].apply(preprocess)

tokenized_sentences = df["tokens"].tolist()

word2vec_model = Word2Vec(
    sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4
)


def get_average_word2vec(tokens, model, vector_size):
    word_vectors = [model.wv[word] for word in tokens if word in model.wv]

    if len(word_vectors) == 0:
        return np.zeros(vector_size)

    return np.mean(word_vectors, axis=0)


df["word2vec_vector"] = df["tokens"].apply(
    lambda tokens: get_average_word2vec(tokens, word2vec_model, 100)
)


def recommend_category_with_word2vec(custom_profile, df):
    custom_profile_text = f"education : {custom_profile['education']} experience : {custom_profile['experience']} skills : {custom_profile['skills']}"
    custom_profile_tokens = preprocess(custom_profile_text)
    custom_profile_vector = get_average_word2vec(
        custom_profile_tokens, word2vec_model, 100
    )

    similarity_scores = cosine_similarity(
        [custom_profile_vector], list(df["word2vec_vector"].values)
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

recommended_category, similarity_score = recommend_category_with_word2vec(
    custom_tech_profile, df
)

print(
    f"Recommended Category: {recommended_category} (Similarity Score: {similarity_score:.2f})"
)
