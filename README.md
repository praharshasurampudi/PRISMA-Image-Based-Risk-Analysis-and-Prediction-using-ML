# PRISMA 
## Image-Based Risk Analysis and Prediction using ML

**PRISMA** is an end-to-end machine learning pipeline for **medical image–based risk prediction**. The system combines **deep learning–derived image embeddings** with **structured tabular data** to predict patient risk levels, with a strong emphasis on **recall** and **ROC-AUC**, which are critical metrics in clinical screening and risk stratification scenarios. This repository demonstrates a realistic, production-style workflow that integrates **PyTorch** for image feature extraction and **Scikit-learn** for classical machine learning, evaluation, and decision optimization.

---

## Key Objectives

* Extract semantic representations from medical images using **pretrained CNNs**
* Combine image-derived features with structured patient metadata
* Train and compare multiple classification models for risk prediction
* Evaluate models using **ROC-AUC** and **Recall**
* Tune decision thresholds to **maximize recall** for high-risk cases

---

## Technology Stack

* **Python 3.12**
* **PyTorch & torchvision** — pretrained ResNet-50 for image embeddings
* **Scikit-learn** — classical ML models, evaluation, threshold tuning
* **Pandas & NumPy** — data handling and feature engineering
* **SQLAlchemy (design-level usage)** — structured data abstraction
* **KaggleHub** — reproducible dataset download

---

## Dataset

### Source

* Public medical imaging dataset from Kaggle:
  **COVID‑19 / Pneumonia / Normal Chest X‑Ray (PA View)**

### Dataset Characteristics

* ~6,900 chest X-ray images
* Three clinical categories:

  * `normal` → Low Risk (0)
  * `covid` → High Risk (1)
  * `pneumonia` → High Risk (1)

### Risk Definition

Risk is defined as the **likelihood that a patient requires further clinical attention**.

---

## Project Workflow

### 1. Environment Setup

* Python virtual environment created
* Dependencies installed via `requirements.txt`

### 2. Dataset Acquisition

* Dataset downloaded programmatically using **kagglehub** (≈1.9 GB)
* Images accessed directly from the local KaggleHub cache

### 3. Structured Data Construction

Since real EHR metadata is unavailable, **synthetic but realistic structured features** were generated:

* `age`
* `bmi`
* `prior_conditions`
* `risk_label`

These features simulate patient-level context typically available in clinical systems.

### 4. Image Embedding Extraction (PyTorch)

* Pretrained **ResNet‑50** loaded from `torchvision`
* Classification head removed
* Each image passed through the network in inference mode
* Output: **2048‑dimensional embedding per image**

This step is the primary usage of **PyTorch** in the project.

### 5. Feature Engineering

* Structured features concatenated with image embeddings
* Final feature matrix shape: **(6902, 2051)**

### 6. Model Training (Scikit-learn)

Four classification models were trained:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. Support Vector Machine (Linear)

All models were trained using **stratified train/test splits** and class balancing where applicable.

### 7. Model Evaluation

Models were evaluated using:

* **ROC-AUC** — ranking quality
* **Recall** — ability to capture high-risk cases

#### Evaluation Results

| Model               | ROC-AUC | Recall |
| ------------------- | ------- | ------ |
| Logistic Regression | 0.9804  | 0.9412 |
| Random Forest       | 0.9716  | 0.9488 |
| Gradient Boosting   | 0.9737  | 0.9499 |
| SVM                 | 0.9757  | 0.9455 |

These results demonstrate strong discriminative power across all models.

### 8. Decision Threshold Tuning

To further improve clinical safety:

* Decision thresholds were swept from **0.10 → 0.90**
* Recall was measured at each threshold

**Best recall achieved:**

* Threshold = **0.10**
* Recall ≈ **0.984**

This shows how high ROC-AUC enables recall optimization without retraining models.

---

## Repository Structure

```
PRISMA/
├── data/
│   ├── structured_data.csv
│   ├── X_features.csv
│   └── y_labels.csv
├── embeddings/
│   └── image_embeddings.csv
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   └── svm.pkl
├── src/
│   ├── download_dataset.py
│   ├── inspect_dataset.py
│   ├── build_structured_dataset.py
│   ├── extract_embeddings.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_models.py
│   ├── threshold_tuning.py
│   ├── load_data.py
└── README.md
└── requirements.txt
```

---

## 📝 License © [Praharsha Surampudi](https://www.linkedin.com/in/praharsha-surampudi/)
This project is for **educational and portfolio purposes only**. It does **not** provide medical advice or diagnostic capabilities.

