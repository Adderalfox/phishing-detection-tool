# 🛡️ Phishing Detection Tool

This project presents a character-level Convolutional Neural Network for detecting phishing URLs. It leverages deep learning to automatically learn patterns from raw character input — enabling robust and scalable phishing detection without manual feature engineering.

---

## 📊 Dataset

**[PhishIUL - Phishing URL Dataset (UCI)](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)**  
- 45,000+ URLs labeled as *phishing* or *legitimate*  
- Features include domain name structure, URL length, entropy, and more  
- Preprocessing includes character-level tokenization for CNN input

---

## 🧠 Model Overview

A robust **CNN + BiLSTM** hybrid model built with **PyTorch** for URL-based phishing detection:

- **Embedding Layer**  
  Transforms input tokens into dense 256-dimensional vectors

- **2 Convolutional Blocks**  
  - `Conv1d → ReLU → MaxPool` structure  
  - First block uses a kernel size of 5, the second uses 3  
  - Captures both broad and fine-grained character-level patterns

- **Bidirectional LSTM**  
  - Stacked 2-layer **BiLSTM** with 256 hidden units  
  - Captures contextual dependencies in both forward and backward directions

- **Fully Connected Layers**  
  - Dense layers with **ReLU** and **Dropout (0.4)** for regularization  
  - Final output: Binary classification — **Phishing** or **Legitimate**

This architecture strikes a balance between simplicity and performance — making it ideal for real-time deployment or browser-based extensions.

---

## 🚀 Key Highlights

- 📌 **Character-Level Input**: Works directly on raw URLs — no manual feature extraction
- ⚙️ **Built from Scratch**: CNN+LSTM implementation without pre-trained models
- 📈 **High Accuracy**: Achieved strong classification performance on a real-world dataset
- 🔬 **Interpretable and Extensible**: Clean design ready for future integration with NLP or hybrid models

---

## 🛠️ Tech Stack

- **Language**: Python  
- **Framework**: PyTorch  
- **Tools**: NumPy, Pandas, Matplotlib, Scikit-learn

---

## 📂 Repository Overview

- `/notebooks` – Notebook for experimentation
- `/src/model.py` – PhishModel architecture  
- `/src/train.py` – Training and evaluation pipeline  
- `/src/preprocess.py` – Preprocessing the data
- `data/` – Dataset preprocessing and loaders  
- `artifacts/` – Stores the Meta-data of preprocessing

---

## 👨‍💻 Author

**Anuranan Chetia**  
Deep Learning | Cybersecurity | AI Research  
[GitHub](https://github.com/Adderalfox) • [LinkedIn](https://www.linkedin.com/in/anuranan-chetia-74452428a/)

---
