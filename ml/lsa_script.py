import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

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

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")

# Fit and transform the combined text data
tfidf_matrix = tfidf_vectorizer.fit_transform(df["combined_text"])

# Apply Latent Semantic Analysis (LSA) using TruncatedSVD
n_components = 5  # Number of topics/components to extract
lsa = TruncatedSVD(n_components=n_components)
lsa_matrix = lsa.fit_transform(tfidf_matrix)


# Function to recommend the closest category using LSA and Cosine Similarity
def recommend_category_with_lsa(custom_profile, df, lsa_matrix, n_components):
    # Combine the custom profile into a single text field
    custom_profile_text = f"education : {custom_profile['education']} experience : {custom_profile['experience']} skills : {custom_profile['skills']}"

    # Vectorize the custom profile using the same TF-IDF vectorizer
    custom_profile_vector = tfidf_vectorizer.transform([custom_profile_text])

    # Transform the custom profile using the fitted LSA model
    custom_profile_lsa = lsa.transform(custom_profile_vector)

    # Compute cosine similarity between the custom profile (in LSA space) and all other profiles (in LSA space)
    similarity_scores = cosine_similarity(custom_profile_lsa, lsa_matrix).flatten()

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

# Get the recommended category based on the custom profile using LSA and cosine similarity
recommended_category, similarity_score = recommend_category_with_lsa(
    custom_tech_profile, df, lsa_matrix, n_components
)

# Display the result
print(
    f"Recommended Category: {recommended_category} (Similarity Score: {similarity_score:.2f})"
)
