import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
import nltk
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout,
)
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

# Prepare data for CNN
df["text"] = df["tokens"].apply(lambda tokens: " ".join(tokens))

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"])

# Prepare embedding matrix
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if word in word2vec_model.wv:
        embedding_vector = word2vec_model.wv[word]
        embedding_matrix[i] = embedding_vector

# Pad sequences
max_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_length)

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

# Define CNN model
model = Sequential()
model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=max_length,
        trainable=False,
    )
)
model.add(Conv1D(filters=128, kernel_size=5, activation="relu"))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(label_encoder.classes_), activation="softmax"))

model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
)

# Evaluate model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

accuracy = np.mean(y_pred_classes == y_test)
print(f"Accuracy: {accuracy:.2f}")

# Classification report
target_names = label_encoder.classes_
report = classification_report(y_test, y_pred_classes, target_names=target_names)

with open("accuracy/accuracy_report_cnn.txt", "w") as f:
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


def predict_category_cnn(custom_profile, tokenizer, model, label_encoder, max_length):
    custom_text = (
        f"education: {custom_profile['education']} "
        f"experience: {custom_profile['experience']} "
        f"skills: {custom_profile['skills']}"
    )
    custom_tokens = preprocess(custom_text)
    custom_sequence = tokenizer.texts_to_sequences([" ".join(custom_tokens)])
    custom_padded = pad_sequences(custom_sequence, maxlen=max_length)
    pred = model.predict(custom_padded)
    pred_class = np.argmax(pred, axis=1)
    category = label_encoder.inverse_transform(pred_class)
    return category[0]


# Predict categories for test profiles
for i, profile in enumerate(test_profiles):
    predicted_category = predict_category_cnn(
        profile, tokenizer, model, label_encoder, max_length
    )
    print(f"Test Profile {i + 1} - Predicted Category: {predicted_category}")
