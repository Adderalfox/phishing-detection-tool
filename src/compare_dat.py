import os
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score

from model import PhishModel
from preprocess import preprocess_dataset
from train import PhishingDataset
from evaluate import evaluate_model

# ========== Config ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
DATA_DIR = "../data/"
MODEL_DIR = "../models/"
GRAPH_DIR = "../graphs/"
ARTIFACTS_DIR = "../artifacts/"

# ========== Create Graphs Directory ==========
os.makedirs(GRAPH_DIR, exist_ok=True)

# ========== Plot Confusion Matrix ==========
def plot_confusion_matrix(cm, labels, dataset_no):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - Dataset {dataset_no}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, f'conf_matrix_dataset{dataset_no}.png'))
    plt.close()

# ========== Plot Confusion Matrix ==========
def plot_confusion_matrix(cm, labels, dataset_no):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - Dataset {dataset_no}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, f'conf_matrix_dataset{dataset_no}.png'))
    plt.close()

# ========== Comparison Plot ==========
def plot_comparison_graph(metrics):
    datasets = [m['dataset'] for m in metrics]
    accuracies = [m['accuracy'] for m in metrics]
    precisions = [m['precision'] for m in metrics]
    recalls = [m['recall'] for m in metrics]

    x = range(len(datasets))

    plt.figure(figsize=(10, 6))
    plt.bar(x, accuracies, width=0.25, label='Accuracy')
    plt.bar([i + 0.25 for i in x], precisions, width=0.25, label='Precision')
    plt.bar([i + 0.5 for i in x], recalls, width=0.25, label='Recall')

    plt.xticks([i + 0.25 for i in x], datasets)
    plt.ylabel('Score')
    plt.title('CNN+LSTM Model Comparison Across Datasets')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, 'model_comparison.png'))
    plt.close()

# ========== Main ==========
def main():
    metrics = []

    for i in range(1, 5):
        print(f"\n=== Evaluating Model {i} ===")
        dataset_path = os.path.join(DATA_DIR, f"Phishing_URL_Dataset_{i}.csv")
        model_path = os.path.join(MODEL_DIR, f"best_model_{i}.pt")
        artifact_path = os.path.join(ARTIFACTS_DIR, f"preprocess_meta_{i}.json")

        # Preprocess
        _, X_val, _, y_val = preprocess_dataset(dataset_path, model_num=i, save_path=ARTIFACTS_DIR)
        with open(artifact_path) as f:
            meta = json.load(f)
        vocab_size = len(meta['char2idx'])

        val_dataset = PhishingDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        print(f'Loading model {model_path}')

        # Load Model
        model = PhishModel(vocab_size).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))

        # Evaluate
        y_true, y_pred = evaluate_model(model, val_loader)

        # Metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        # Save Confusion Matrix
        plot_confusion_matrix(cm, labels=["Benign", "Phishing"], dataset_no=i)

        DATASET_NAMES = {
            1: "data_bal-20000.xlsx",
            2: "malicious_phish_filtered_2.csv",
            3: "phishing_site_urls_3.csv",
            4: "Phishing_URL_Dataset_4.csv"
        }


        # Store for summary graph
        metrics.append({
            'dataset': DATASET_NAMES[i],
            'accuracy': acc,
            'precision': prec,
            'recall': rec
        })

        print(f"\nModel {i} Results:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print("Classification Report:\n", classification_report(y_true, y_pred))

    # Plot Overall Comparison
    plot_comparison_graph(metrics)


if __name__ == '__main__':
    main()