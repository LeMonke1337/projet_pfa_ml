import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the dataset
df = pd.read_csv("resumme_v3.csv")

# Combine text fields (education, experience, skills) into a single field
df["combined_text"] = (
    df["education"].fillna("")
    + " "
    + df["experience"].fillna("")
    + " "
    + df["skills"].fillna("")
)

# Step 1: TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(df["combined_text"])

# Step 2: LSA dimensionality reduction
n_components = 100  # You can adjust the number of components
lsa = TruncatedSVD(n_components=n_components)
lsa_matrix = lsa.fit_transform(tfidf_matrix)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    lsa_matrix, df["category"], test_size=0.2, random_state=42
)

# Step 4: SVM Classifier
svm_classifier = SVC(kernel="linear", random_state=42)
svm_classifier.fit(X_train, y_train)

# Step 5: Predictions and evaluation
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save accuracy and classification report to a file
with open("accuracy/lsa_svm_accuracy.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.2f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))

print(f"SVM Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# 10 Custom profiles to test
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


# Function to recommend category using SVM with LSA features
def recommend_category_with_svm(profile, tfidf_vectorizer, lsa, svm_classifier):
    profile_text = (
        f"education: {profile['education']} "
        f"experience: {profile['experience']} "
        f"skills: {profile['skills']}"
    )

    # TF-IDF transform of the new profile
    profile_tfidf = tfidf_vectorizer.transform([profile_text])

    # LSA transformation
    profile_lsa = lsa.transform(profile_tfidf)

    # SVM prediction
    predicted_category = svm_classifier.predict(profile_lsa)[0]

    return predicted_category


# Test on 10 custom profiles
for i, profile in enumerate(test_profiles):
    recommended_category = recommend_category_with_svm(
        profile, tfidf_vectorizer, lsa, svm_classifier
    )
    print(f"Test Profile {i + 1} - Recommended Category: {recommended_category}")
