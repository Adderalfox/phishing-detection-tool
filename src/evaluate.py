import os
import torch
import json
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from model import PhishModel
from preprocess import preprocess_dataset
from train import PhishingDataset
from tqdm import tqdm

# ========== Config ==========
CSV_PATH = '../data/Phishing_URL_Dataset_3.csv'
ARTIFACTS_PATH = '../artifacts/'
MODEL_PATH = '../models/best_model_3.pt'
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== Evaluate Function ==========
def evaluate_model(model, dataloader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return y_true, y_pred

# ========== Main Function ==========
def main():
    print("Loading data...")
    _, X_val, _, y_val = preprocess_dataset(CSV_PATH, save_path=ARTIFACTS_PATH)

    with open(os.path.join(ARTIFACTS_PATH, 'preprocess_meta_3.json')) as f:
        meta = json.load(f)
    vocab_size = len(meta['char2idx'])

    val_dataset = PhishingDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    print("Loading Model...")
    model = PhishModel(vocab_size).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))

    print("Running evaluation...")
    y_true, y_pred = evaluate_model(model, val_loader)

    print('\nClassification Report: ')
    print(classification_report(y_true, y_pred))

    print('\nConfusion Matrix: ')
    print(confusion_matrix(y_true, y_pred))

if __name__ == '__main__':
    main()
