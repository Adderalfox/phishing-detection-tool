import torch
import json
import os
from model import PhishModel
from preprocess import encode_url

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_meta(meta_path: str):
    with open(meta_path) as f:
        return json.load(f)
    
def load_model(model_path: str, vocab_size: int):
    model = PhishModel(vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model
def predict_url(model, url: str, char2idx: dict, maxlen: int = 200):
    encoded = encode_url(url, char2idx, maxlen= maxlen)
    input_tensor = torch.tensor([encoded], dtype=torch.long).to(DEVICE)
    with torch.zero_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item()
    return prediction, confidence