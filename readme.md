# 🛡️ Phishing Detection Using Vanilla CNN (CharCNN)

This project presents a character-level Convolutional Neural Network (CharCNN) for detecting phishing URLs. It leverages deep learning to automatically learn patterns from raw character input — enabling robust and scalable phishing detection without manual feature engineering.

---

## 📊 Dataset

**[PhishIUL - Phishing URL Dataset (UCI)](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)**  
- 45,000+ URLs labeled as *phishing* or *legitimate*  
- Features include domain name structure, URL length, entropy, and more  
- Preprocessing includes character-level tokenization for CNN input

---

## 🧠 Model Overview

A lightweight **Vanilla CNN** built using **PyTorch**, optimized for sequence classification:

- **Embedding Layer**: Converts characters into dense vectors  
- **2 Convolutional Blocks**: Detects character-level patterns in URL strings  
- **Fully Connected Layers**: Outputs binary classification (phishing or not)  
- Regularized using **Dropout**, activated with **ReLU**

This architecture strikes a balance between simplicity and performance — making it ideal for real-time deployment or browser-based extensions.

---

## 🚀 Key Highlights

- 📌 **Character-Level Input**: Works directly on raw URLs — no manual feature extraction
- ⚙️ **Built from Scratch**: Vanilla CNN implementation without pre-trained models
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
- `/src/model.py` – CharCNN architecture  
- `/src/train.py` – Training and evaluation pipeline  
- `/src/preprocess.py` – Preprocessing the data
- `data/` – Dataset preprocessing and loaders  
- `artifacts/` – Stores the Meta-data of preprocessing

---

## 👨‍💻 Author

**Anuranan Chetia**  
Deep Learning | Cybersecurity | AI Research  
[GitHub](https://github.com/Adderalfox) • [LinkedIn](www.linkedin.com/in/anuranan-chetia-74452428a)

---