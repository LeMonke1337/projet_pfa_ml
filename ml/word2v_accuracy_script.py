import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.tokenize import word_tokenize
import numpy as np
import nltk
from sklearn.metrics import classification_report

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


correct_predictions = 0
total_predictions = len(df)
y_true = []
y_pred = []

for index, row in df.iterrows():
    custom_profile = {
        "education": row["education"],
        "experience": row["experience"],
        "skills": row["skills"],
    }

    recommended_category, _ = recommend_category_with_word2vec(custom_profile, df)

    y_true.append(row["category"])
    y_pred.append(recommended_category)

    if recommended_category == row["category"]:
        correct_predictions += 1

accuracy = correct_predictions / total_predictions
report = classification_report(y_true, y_pred, output_dict=True)

with open("accuracy/accuracy_report.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.2f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_true, y_pred))

test_profiles = [
    {
        "education": "Master's in Software Engineering",
        "experience": "3 years in machine learning and data analysis.",
        "skills": "Python, TensorFlow, SQL, Data Visualization",
    },
    {
        "education": "Bachelor's in Information Technology",
        "experience": "5 years in IT project management.",
        "skills": "Agile, Scrum, Java, Cloud Computing",
    },
    {
        "education": "PhD in Computer Science",
        "experience": "2 years as a research scientist.",
        "skills": "Machine Learning, Python, R, Research",
    },
    {
        "education": "Bachelor's in Graphic Design",
        "experience": "4 years in web design and UX/UI.",
        "skills": "Adobe Photoshop, Figma, HTML, CSS",
    },
    {
        "education": "Master's in Cybersecurity",
        "experience": "5 years in network security.",
        "skills": "Firewalls, Intrusion Detection, Penetration Testing",
    },
    {
        "education": "Bachelor's in Data Science",
        "experience": "2 years as a data analyst.",
        "skills": "R, Python, SQL, Machine Learning",
    },
    {
        "education": "Master's in Business Administration",
        "experience": "6 years in marketing management.",
        "skills": "SEO, Content Strategy, Data Analysis",
    },
    {
        "education": "Bachelor's in Electrical Engineering",
        "experience": "4 years in hardware development.",
        "skills": "Circuit Design, MATLAB, Robotics",
    },
    {
        "education": "Bachelor's in Marketing",
        "experience": "3 years in digital marketing.",
        "skills": "Social Media, Google Ads, Analytics",
    },
    {
        "education": "Master's in Artificial Intelligence",
        "experience": "2 years in AI research and development.",
        "skills": "Deep Learning, Python, Neural Networks",
    },
]

for i, profile in enumerate(test_profiles):
    recommended_category, similarity_score = recommend_category_with_word2vec(
        profile, df
    )
    print(
        f"Test Profile {i + 1} - Recommended Category: {recommended_category} (Similarity Score: {similarity_score:.2f})"
    )
