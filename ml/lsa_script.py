import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("resumme_v3.csv")

df["combined_text"] = (
    df["education"].fillna("")
    + " "
    + df["experience"].fillna("")
    + " "
    + df["skills"].fillna("")
)

tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")

tfidf_matrix = tfidf_vectorizer.fit_transform(df["combined_text"])

n_components = 5
lsa = TruncatedSVD(n_components=n_components)
lsa_matrix = lsa.fit_transform(tfidf_matrix)


def recommend_category_with_lsa(custom_profile, df, lsa_matrix, n_components):
    custom_profile_text = f"education : {custom_profile['education']} experience : {custom_profile['experience']} skills : {custom_profile['skills']}"

    custom_profile_vector = tfidf_vectorizer.transform([custom_profile_text])

    custom_profile_lsa = lsa.transform(custom_profile_vector)

    similarity_scores = cosine_similarity(custom_profile_lsa, lsa_matrix).flatten()

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

recommended_category, similarity_score = recommend_category_with_lsa(
    custom_tech_profile, df, lsa_matrix, n_components
)

print(
    f"Recommended Category: {recommended_category} (Similarity Score: {similarity_score:.2f})"
)
