import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("resumme_v3.csv")

text_data = df["text"].fillna("")
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")

tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)


def recommend_category(custom_profile, df):
    custom_profile_text = f"education : {custom_profile['education']} experience : {custom_profile['experience']} skills : {custom_profile['skills']}"

    custom_profile_vector = tfidf_vectorizer.transform([custom_profile_text])

    similarity_scores = cosine_similarity(custom_profile_vector, tfidf_matrix).flatten()

    df["similarity_score"] = similarity_scores

    category_similarity = (
        df.groupby("category")["similarity_score"].mean().reset_index()
    )

    top_category = category_similarity.sort_values(
        by="similarity_score", ascending=False
    ).iloc[0]

    return top_category["category"], top_category["similarity_score"]


custom_profile = {
    "education": "Bachelor of Science in Computer Science",
    "experience": "4 years of experience as a software engineer, focusing on web development, cloud computing, and microservices architecture. Led a team in developing scalable web applications and integrated cloud-based solutions.",
    "skills": "Python, Java, JavaScript, AWS, Docker, Kubernetes, CI/CD, REST APIs, Agile methodologies",
}

recommended_category, similarity_score = recommend_category(custom_profile, df)

print(
    f"Recommended Category: {recommended_category} (Similarity Score: {similarity_score:.2f})"
)
