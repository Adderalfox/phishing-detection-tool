import pandas as pd
import re
import json
import os
from sklearn.model_selection import train_test_split

def clean_url(url):
    url = url.lower()
    url = re.sub(r'https?://', '', url)
    url = re.sub(r'www\.', '', url)
    return url.strip()

def build_vocab(urls):
    all_text = ''.join(urls)
    vocab = sorted(set(all_text))
    char2idx = {ch: idx + 1 for idx, ch in enumerate(vocab)}
    idx2char = {idx: ch for ch, idx in char2idx.items()}
    return char2idx, idx2char

def encode_url(url, char2idx, maxlen=200):
    encoded = [char2idx.get(c, 0) for c in url[:maxlen]]
    if len(encoded) < maxlen:
        encoded += [0] * (maxlen - len(encoded))
    return encoded

def preprocess_dataset(csv_path, model_num, save_path='../artifacts/', maxlen=200, test_size=0.2):
    df = pd.read_csv(csv_path)
    df = df[['URL', 'label']]
    # df['label'] = df['label'].apply(lambda x: 0 if x == 1 else 1)
    df['clean_url'] = df['URL'].apply(clean_url)

    char2idx, idx2char = build_vocab(df['clean_url'].tolist())

    df['encoded_url'] = df['clean_url'].apply(lambda x: encode_url(x, char2idx, maxlen))

    X_train, X_val, y_train, y_val = train_test_split(
        df['encoded_url'].tolist(), df['label'].tolist(), test_size=test_size, random_state=42
    )

    os.makedirs(save_path, exist_ok=True)
    
    with open(os.path.join(save_path, f'preprocess_meta_{model_num}.json'), 'w') as f:
        json.dump({'char2idx': char2idx, 'maxlen': maxlen}, f)
    
    print('Preprocessing complete.')

    return X_train, X_val, y_train, y_val
