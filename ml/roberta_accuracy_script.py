import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import os
import pickle


csv_path = "resumme_v3.csv"

if os.path.exists(csv_path):
    print("File exists")
else:
    print("File does not exist")

df = pd.read_csv(csv_path)


if "text" not in df.columns:
    df["text"] = (
        df["education"].fillna("")
        + " "
        + df["experience"].fillna("")
        + " "
        + df["skills"].fillna("")
    )
else:
    df["text"] = df["text"].astype(str)


tokenizer = RobertaTokenizer.from_pretrained("./tokenizer_roberta_v1")


with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


def preprocess_texts(texts, tokenizer, max_length=512):
    encodings = tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        return_tensors="tf",
        max_length=max_length,
    )
    return encodings


texts = df["text"]
encodings = preprocess_texts(texts, tokenizer)


if "label" in df.columns:
    label_column = "label"
elif "category" in df.columns:
    label_column = "category"
else:
    raise ValueError(
        "No label column found in DataFrame. Please ensure 'label' or 'category' column exists."
    )

labels = label_encoder.transform(df[label_column])

model = TFRobertaForSequenceClassification.from_pretrained("./model_roberta_v1")

batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices((dict(encodings)))
dataset = dataset.batch(batch_size)

predictions = []
for batch in dataset:
    outputs = model(**batch, training=False)
    logits = outputs.logits
    batch_predictions = tf.argmax(logits, axis=1).numpy()
    predictions.extend(batch_predictions)

predicted_labels = np.array(predictions)

accuracy = np.mean(predicted_labels == labels)

target_names = [str(cls) for cls in label_encoder.classes_]

report = classification_report(labels, predicted_labels, target_names=target_names)

with open("accuracy/accuracy_report_roberta.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.2f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

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


def predict_category(custom_profile, model, tokenizer, label_encoder, max_length=512):
    custom_text = f"education: {custom_profile['education']} experience: {custom_profile['experience']} skills: {custom_profile['skills']}"
    encoding = tokenizer.encode_plus(
        custom_text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="tf",
    )
    inputs = {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
    }
    outputs = model(**inputs, training=False)
    logits = outputs.logits
    predicted_label = tf.argmax(logits, axis=1).numpy()
    predicted_category = label_encoder.inverse_transform(predicted_label)

    return predicted_category[0]


for i, profile in enumerate(test_profiles):
    predicted_category = predict_category(profile, model, tokenizer, label_encoder)
    print(f"Test Profile {i + 1} - Predicted Category: {predicted_category}")
