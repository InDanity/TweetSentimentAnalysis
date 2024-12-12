import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, auc
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from datasets import load_dataset
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_auc(precision, recall):
    # To calculate AUC, we need to sort precision and recall
    sorted_indices = np.argsort(recall)
    sorted_recall = np.array(recall)[sorted_indices]
    sorted_precision = np.array(precision)[sorted_indices]

    # Calculate AUC using trapezoidal rule (area under Precision-Recall curve)
    auc_score = auc(sorted_recall, sorted_precision)
    print(f"AUC (based on Precision-Recall curve): {auc_score:.4f}")

    return auc_score


def calculate_metrics(conf_matrix_numpy_array):
    conf_matrix = pd.DataFrame(data=conf_matrix_numpy_array)

    num_classes = conf_matrix.shape[0]

    # Initialize lists to store precision, recall, IoU, F1
    precision_list = []
    recall_list = []
    iou_list = []
    f1_list = []

    for i in range(num_classes):
        TP = conf_matrix.iloc[i, i]  # True positives for class i
        FP = conf_matrix.iloc[:, i].sum() - TP  # False positives for class i
        FN = conf_matrix.iloc[i, :].sum() - TP  # False negatives for class i
        TN = conf_matrix.sum().sum() - (TP + FP + FN)  # True negatives (not used directly)

        # Precision, recall, IoU for class i
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

        # F1 Score for class i
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        iou_list.append(iou)
        f1_list.append(f1_score)

        print(f"Class {i}: Precision={precision:.4f}, Recall={recall:.4f}, IoU={iou:.4f}, F1 Score={f1_score:.4f}")

    # Average precision, recall, IoU, and F1 across classes
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_iou = np.mean(iou_list)
    avg_f1 = np.mean(f1_list)

    print(f"\nAverage Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")

    calculate_auc(precision_list, recall_list)

    return precision_list, recall_list, iou_list, f1_list


# Load the tokenizer and model
model_str = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_str)
model = pickle.load(open("model_pickle", 'rb'))

# Load datasets
train_ds = load_dataset("SetFit/tweet_sentiment_extraction", split="train")
test_ds = load_dataset("SetFit/tweet_sentiment_extraction", split="test")

# Prepare data loaders
def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",  # Adds padding to the maximum length
        truncation=True,  # Truncate if length exceeds max_length
        max_length=128,  # Set a reasonable max length (adjust based on your dataset)
        return_tensors="pt"  # Returns PyTorch tensors
    )

train_ds = train_ds.map(tokenize_function, batched=True)
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
train_loader = DataLoader(train_ds, shuffle=True, batch_size=8)  # Adjust batch size as needed

# Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)  # Learning rate can be adjusted
num_epochs = 3  # Number of epochs can be adjusted
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Switch model to evaluation mode
model.eval()

# Evaluate on the test set
true_labels = []
predicted_labels = []

for tweet_data in test_ds:
    tweet = tweet_data['text']
    true_label = tweet_data['label']

    inputs = tokenizer(tweet, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()

    true_labels.append(true_label)
    predicted_labels.append(predicted_class_id)

    print("Tweet: " + tweet)
    print("Predicted Sentiment: " + str(predicted_class_id))
    print("Actual Sentiment: " + str(true_label))
    print()


# Generate the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
labels = list(model.config.id2label.values())

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()

# Calculate and display per-class accuracy
class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
print("Per-Class Accuracy:")
for i, label in enumerate(labels):
    print(f"Accuracy for {label}: {class_accuracy[i]:.2f}")

# Evaluation Results
print("Evaluation Results:")
print(f"Accuracy: {accuracy_score(true_labels, predicted_labels)}")
print("Classification Report:")
print(classification_report(true_labels, predicted_labels))
