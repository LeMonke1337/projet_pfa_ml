import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import pickle

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

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(df["combined_text"])

# LSA Transformation
n_components = 100  # Increased components for better feature representation
lsa = TruncatedSVD(n_components=n_components)
lsa_matrix = lsa.fit_transform(tfidf_matrix)

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df["category"])

# Save the LabelEncoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    lsa_matrix, labels, test_size=0.2, random_state=42, stratify=labels
)

# Reshape data for LSTM input
X_train_lstm = np.expand_dims(X_train, axis=1)
X_test_lstm = np.expand_dims(X_test, axis=1)

# Define LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(1, n_components)))
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu"))
model.add(Dense(len(label_encoder.classes_), activation="softmax"))

model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# Train model
early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

history = model.fit(
    X_train_lstm,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1,
)

# Evaluate model
y_pred_prob = model.predict(X_test_lstm)
y_pred = np.argmax(y_pred_prob, axis=1)

accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")

# Classification report
target_names = label_encoder.classes_
report = classification_report(y_test, y_pred, target_names=target_names)

with open("accuracy/accuracy_report_lsa_lstm.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.2f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("Classification Report:")
print(report)

# Test profiles (including all 10 profiles)
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


def predict_category_lsa_lstm(
    custom_profile, tfidf_vectorizer, lsa, model, label_encoder
):
    custom_profile_text = (
        f"education: {custom_profile['education']} "
        f"experience: {custom_profile['experience']} "
        f"skills: {custom_profile['skills']}"
    )
    custom_profile_vector = tfidf_vectorizer.transform([custom_profile_text])
    custom_profile_lsa = lsa.transform(custom_profile_vector)
    custom_profile_lsa_lstm = np.expand_dims(custom_profile_lsa, axis=1)
    pred_prob = model.predict(custom_profile_lsa_lstm)
    pred_class = np.argmax(pred_prob, axis=1)
    category = label_encoder.inverse_transform(pred_class)
    return category[0]


# Predict categories for test profiles
print("\nPredictions for Test Profiles:")
for i, profile in enumerate(test_profiles):
    predicted_category = predict_category_lsa_lstm(
        profile, tfidf_vectorizer, lsa, model, label_encoder
    )
    print(f"Test Profile {i + 1} - Predicted Category: {predicted_category}")
