import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import AlbertTokenizer, TFAlbertForSequenceClassification
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import pickle

csv_path = "resumme_v3.csv"

if os.path.exists(csv_path):
    print("File exists")
else:
    print("File does not exist")

df = pd.read_csv(csv_path)
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
df["text"] = df["text"].astype(str)

labels = df["label"]

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"], labels_encoded, test_size=0.2, stratify=labels_encoded
)

train_encodings = tokenizer(
    train_texts.tolist(), truncation=True, padding=True, max_length=512
)
val_encodings = tokenizer(
    val_texts.tolist(), truncation=True, padding=True, max_length=512
)

train_dataset = tf.data.Dataset.from_tensor_slices(
    (dict(train_encodings), train_labels)
)
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels))

num_labels = len(label_encoder.classes_)

model = TFAlbertForSequenceClassification.from_pretrained(
    "albert-base-v2", num_labels=num_labels
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(
    train_dataset.shuffle(1000).batch(16),
    validation_data=val_dataset.batch(64),
    epochs=6,
)

model.save_pretrained("./model_albert_v2")
tokenizer.save_pretrained("./tokenizer_albert_v2")
