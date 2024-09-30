import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
import nltk
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pickle

nltk.download("punkt")

# Load the dataset
df = pd.read_csv("resumme_v3.csv")

# Combine 'education', 'experience', and 'skills' into 'combined_text'
df["combined_text"] = (
    df["education"].fillna("")
    + " "
    + df["experience"].fillna("")
    + " "
    + df["skills"].fillna("")
)


# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    return tokens


# Apply preprocessing
df["tokens"] = df["combined_text"].apply(preprocess)

# Prepare data for Word2Vec
tokenized_sentences = df["tokens"].tolist()

# Train Word2Vec model
word2vec_model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
)


# Function to get average Word2Vec vectors
def get_average_word2vec(tokens_list, model, vector_size):
    vectors = []
    for tokens in tokens_list:
        word_vectors = [model.wv[word] for word in tokens if word in model.wv]
        if len(word_vectors) > 0:
            vectors.append(np.mean(word_vectors, axis=0))
        else:
            vectors.append(np.zeros(vector_size))
    return vectors


# Get average Word2Vec vectors for all documents
vector_size = 100
X = get_average_word2vec(df["tokens"], word2vec_model, vector_size)
X = np.array(X)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["category"])

# Save the LabelEncoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")

# Classification report
target_names = label_encoder.classes_
report = classification_report(y_test, y_pred, target_names=target_names)

with open("accuracy/accuracy_report_word2vec_logreg.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.2f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("Classification Report:")
print(report)

# Test profiles
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


def predict_category_logreg(
    custom_profile, model, word2vec_model, label_encoder, vector_size
):
    custom_text = (
        f"education: {custom_profile['education']} "
        f"experience: {custom_profile['experience']} "
        f"skills: {custom_profile['skills']}"
    )
    custom_tokens = preprocess(custom_text)
    word_vectors = [
        word2vec_model.wv[word] for word in custom_tokens if word in word2vec_model.wv
    ]
    if len(word_vectors) > 0:
        custom_vector = np.mean(word_vectors, axis=0)
    else:
        custom_vector = np.zeros(vector_size)
    pred = model.predict([custom_vector])
    category = label_encoder.inverse_transform(pred)
    return category[0]


# Predict categories for test profiles
print("\nPredictions for Test Profiles:")
for i, profile in enumerate(test_profiles):
    predicted_category = predict_category_logreg(
        profile, model, word2vec_model, label_encoder, vector_size
    )
    print(f"Test Profile {i + 1} - Predicted Category: {predicted_category}")
