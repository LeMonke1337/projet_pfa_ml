import pandas as pd
import numpy as np
import re
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import tensorflow as tf
from transformers import AlbertTokenizer, TFAlbertForSequenceClassification
import pickle
import warnings

warnings.filterwarnings("ignore")
nltk.download("punkt")

# Load the dataset
csv_path = "resumme_v3.csv"
df = pd.read_csv(csv_path)

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
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Apply preprocessing
df["text"] = df["combined_text"].apply(preprocess)

# Encode labels
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["category"])
num_labels = len(label_encoder.classes_)

# Save the LabelEncoder for consistency
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Split data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# Load your saved tokenizer
tokenizer = AlbertTokenizer.from_pretrained("./tokenizer_albert_v1")

# Tokenize the data
max_length = 512


def tokenize_texts(texts):
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="tf",
    )


train_encodings = tokenize_texts(X_train)
test_encodings = tokenize_texts(X_test)

# Convert labels to tensors
train_labels = tf.convert_to_tensor(y_train.values)
test_labels = tf.convert_to_tensor(y_test.values)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices(
    (dict(train_encodings), train_labels)
)

test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_labels))

batch_size = 16
train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# Load your saved model
model = TFAlbertForSequenceClassification.from_pretrained(
    "./model_albert_v1", num_labels=num_labels
)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ["accuracy"]
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Evaluate the model
results = model.evaluate(test_dataset)
print(f"Test Loss: {results[0]:.2f}, Test Accuracy: {results[1]:.2f}")

# Get predictions
y_preds = []
y_trues = []

for batch in test_dataset:
    inputs, labels = batch
    logits = model(inputs, training=False).logits
    predictions = tf.argmax(logits, axis=-1)
    y_preds.extend(predictions.numpy())
    y_trues.extend(labels.numpy())

# Classification Report
target_names = label_encoder.classes_
report = classification_report(y_trues, y_preds, target_names=target_names)

with open("accuracy/alber_acc.txt", "w") as f:
    f.write(f"Test Accuracy: {results[1]:.2f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("Classification Report:")
print(report)

# Test Profiles
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


def preprocess_custom_text(custom_profile):
    custom_text = (
        f"education: {custom_profile['education']} "
        f"experience: {custom_profile['experience']} "
        f"skills: {custom_profile['skills']}"
    )
    custom_text = preprocess(custom_text)
    return custom_text


def predict_category_albert(custom_profile, tokenizer, model, label_encoder):
    custom_text = preprocess_custom_text(custom_profile)
    inputs = tokenizer(
        [custom_text],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="tf",
    )
    logits = model(inputs, training=False).logits
    prediction = tf.argmax(logits, axis=-1).numpy()
    category = label_encoder.inverse_transform(prediction)
    return category[0]


# Predict categories for test profiles
print("\nPredictions for Test Profiles:")
for i, profile in enumerate(test_profiles):
    predicted_category = predict_category_albert(
        profile, tokenizer, model, label_encoder
    )
    print(f"Test Profile {i + 1} - Predicted Category: {predicted_category}")
