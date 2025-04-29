import os
import requests
from sklearn.metrics import accuracy_score

API_URL = "http://localhost:8000/predict"
TEST_DIR = "./test_images"

# 👇 Map filenames to their true labels
true_labels = {
    "true.jpg": "Benign",
    "false.jpg": "Malignant"
}

predictions = []
ground_truths = []

for filename, true_label in true_labels.items():
    file_path = os.path.join(TEST_DIR, filename)

    with open(file_path, 'rb') as f:
        response = requests.post(API_URL, files={"file": f})

    try:
        result = response.json()

        if "label" not in result:
            print(f"❌ Error for {filename}: {result}")
            continue

        predicted_label = result["label"]
        confidence = result["confidence_score"]

        predictions.append(predicted_label)
        ground_truths.append(true_label)

        print(f"{filename}: predicted = {predicted_label}, true = {true_label}, confidence = {confidence}")

    except Exception as e:
        print(f"❌ Failed to process {filename}: {e}")

# 🧠 Accuracy
if predictions:
    accuracy = accuracy_score(ground_truths, predictions)
    print("\n✅ Model Accuracy:", round(accuracy, 3))
else:
    print("❌ No predictions made.")
