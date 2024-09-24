from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import re
from fastapi import HTTPException


async def recommender_model(text):

    tokenizer = BertTokenizer.from_pretrained("services\\routes\model\\tokenizer_v3")
    model = TFBertForSequenceClassification.from_pretrained(
        "services\\routes\\model\\model_v3", num_labels=24
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    inputs = tokenizer(
        text, return_tensors="tf", truncation=True, padding=True, max_length=128
    )

    predictions = model(inputs)

    probabilities = tf.nn.softmax(predictions.logits, axis=-1).numpy()[0]

    label_mapping = {
        "accountant": 0,
        "advocate": 1,
        "agriculture": 2,
        "apparel": 3,
        "arts": 4,
        "automobile": 5,
        "aviation": 6,
        "banking": 7,
        "bpo": 8,
        "business-development": 9,
        "chef": 10,
        "construction": 11,
        "consultant": 12,
        "designer": 13,
        "digital-media": 14,
        "engineering": 15,
        "finance": 16,
        "fitness": 17,
        "healthcare": 18,
        "hr": 19,
        "information-technology": 20,
        "public-relations": 21,
        "sales": 22,
        "teacher": 23,
    }

    reverse_label_mapping = {v: k for k, v in label_mapping.items()}

    label_probabilities = [
        (reverse_label_mapping[i], prob) for i, prob in enumerate(probabilities)
    ]
    label_probabilities.sort(key=lambda x: x[1], reverse=True)
    result = []
    print("Compatibility Percentages:")
    for label, prob in label_probabilities:
        print(f"{label}: {prob * 100:.2f}%")
        result.append(f"{label}: {prob * 100:.2f}%")
    return result
