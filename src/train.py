import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from preprocess import preprocess_dataset
from model import PhishCNN

# ======== Config ============
CSV_PATH = '../data/Phishing_URL_Dataset.csv'
ARTIFACTS_PATH = '../artifacts/'
MODEL_SAVE_DIR = '../models/'
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')