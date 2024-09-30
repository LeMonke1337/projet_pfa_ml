import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("resumme_v3.csv")

df["combined_text"] = (
    df["education"].fillna("")
    + " "
    + df["experience"].fillna("")
    + " "
    + df["skills"].fillna("")
)

tokenizer = BertTokenizer.from_pretrained("occupation_recommender_tokenizer")


def preprocess_texts(texts, tokenizer, max_length=128):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="tf",
        )

        input_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])

    input_ids = tf.concat(input_ids, axis=0)
    attention_masks = tf.concat(attention_masks, axis=0)

    return input_ids, attention_masks


texts = df["combined_text"].tolist()
input_ids, attention_masks = preprocess_texts(texts, tokenizer)

model = TFBertForSequenceClassification.from_pretrained("occupation_recommender_model")

label_encoder = LabelEncoder()
label_encoder.fit(df["category"])

true_labels = label_encoder.transform(df["category"])

batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_masks))
dataset = dataset.batch(batch_size)

predictions = []
for batch in dataset:
    inputs, masks = batch
    outputs = model(inputs, attention_mask=masks)
    logits = outputs.logits
    batch_predictions = tf.argmax(logits, axis=1).numpy()
    predictions.extend(batch_predictions)

predicted_labels = np.array(predictions)

accuracy = np.mean(predicted_labels == true_labels)
report = classification_report(
    true_labels, predicted_labels, target_names=label_encoder.classes_
)

with open("accuracy_report_bert.txt", "w") as f:
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


def predict_category(custom_profile, model, tokenizer, label_encoder, max_length=128):
    custom_text = f"education: {custom_profile['education']} experience: {custom_profile['experience']} skills: {custom_profile['skills']}"
    encoded_dict = tokenizer.encode_plus(
        custom_text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="tf",
    )
    input_id = encoded_dict["input_ids"]
    attention_mask = encoded_dict["attention_mask"]

    outputs = model(input_id, attention_mask=attention_mask)
    logits = outputs.logits
    predicted_label = tf.argmax(logits, axis=1).numpy()
    predicted_category = label_encoder.inverse_transform(predicted_label)

    return predicted_category[0]


for i, profile in enumerate(test_profiles):
    predicted_category = predict_category(profile, model, tokenizer, label_encoder)
    print(f"Test Profile {i + 1} - Predicted Category: {predicted_category}")
