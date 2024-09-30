import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
import nltk
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


# Load GloVe embeddings
def load_glove_vectors(glove_file):
    glove_dict = {}
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype="float32")
            glove_dict[word] = vector
    return glove_dict


glove_file = "glove.6B.300d.txt"  # Ensure you have this file in the working directory
glove_vectors = load_glove_vectors(glove_file)


# Function to get average GloVe vectors
def get_average_glove(tokens_list, glove_vectors, vector_size):
    vectors = []
    for tokens in tokens_list:
        word_vectors = [glove_vectors[word] for word in tokens if word in glove_vectors]
        if len(word_vectors) > 0:
            vectors.append(np.mean(word_vectors, axis=0))
        else:
            vectors.append(np.zeros(vector_size))
    return vectors


# Get average GloVe vectors for all documents
vector_size = 300
X = get_average_glove(df["tokens"], glove_vectors, vector_size)
X = np.array(X)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["category"])

# Save the LabelEncoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Split data (ensure consistent random_state and stratify)
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

with open("accuracy/accuracy_report_glove_logreg.txt", "w") as f:
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


def predict_category_glove_logreg(
    custom_profile, model, glove_vectors, label_encoder, vector_size
):
    custom_text = (
        f"education: {custom_profile['education']} "
        f"experience: {custom_profile['experience']} "
        f"skills: {custom_profile['skills']}"
    )
    custom_tokens = preprocess(custom_text)
    word_vectors = [
        glove_vectors[word] for word in custom_tokens if word in glove_vectors
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
    predicted_category = predict_category_glove_logreg(
        profile, model, glove_vectors, label_encoder, vector_size
    )
    print(f"Test Profile {i + 1} - Predicted Category: {predicted_category}")
