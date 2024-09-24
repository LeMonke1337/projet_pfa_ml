import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import (
    RobertaTokenizer,
    TFRobertaForSequenceClassification,
)
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os


csv_path = "resumme_v3.csv"

if os.path.exists(csv_path):
    print("File exists")
else:
    print("File does not exist")

df = pd.read_csv(csv_path)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
df["text"] = df["text"].astype(str)
inputs = tokenizer(
    df.text.tolist(), padding=True, truncation=True, return_tensors="tf", max_length=512
)
labels = df.label.tolist()
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"], labels, test_size=0.2
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
model = TFRobertaForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=24
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
model.save_pretrained("./model_roberta_v1")
tokenizer.save_pretrained("./tokenizer_roberta_v1")
