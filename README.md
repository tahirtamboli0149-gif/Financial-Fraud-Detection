# 💳 Financial Transaction Fraud Detection

A machine learning project that detects potentially fraudulent financial transactions using transaction type, amount, and sender balance information.

## 🚀 Project Overview

Financial fraud is a major challenge for digital payment systems. This project uses machine learning to classify transactions as **Normal** or **Fraud**.

The project compares multiple classification models and uses a **Random Forest Classifier** as the final model because it achieved a strong balance between precision, recall, and F1-score on the test data.

A Streamlit web application provides a simple interface where users can enter transaction details and receive a fraud prediction.

## 🧠 Machine Learning

### Input Features

The model uses three transaction features:

* `type` — Transaction type
* `amount` — Transaction amount
* `oldbalanceOrg` — Sender's balance before the transaction

### Target

* `isFraud` — `0` = Normal, `1` = Fraud

### Models Tested

| Model               | Precision | Recall | F1-Score |
| ------------------- | --------: | -----: | -------: |
| Logistic Regression |    0.8014 | 0.1400 |   0.2383 |
| Decision Tree       |    0.7695 | 0.7967 |   0.7829 |
| Random Forest       |    0.8702 | 0.7955 |   0.8312 |

The **Random Forest** model achieved the highest F1-score among the tested models.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Matplotlib
* Seaborn
* Git & GitHub

## 📁 Project Structure

```text
Financial-Fraud-Detection/
│
├── data/
│   └── PaySim dataset
│
├── src/
│   └── training_data.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/tahirtamboli0149-gif/Financial-Fraud-Detection.git
cd Financial-Fraud-Detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

## 🔍 How It Works

```text
Transaction Details
        ↓
Data Preprocessing
        ↓
Random Forest Model
        ↓
Fraud Probability
        ↓
Normal / Fraud Prediction
```

The application also performs a separate balance validation check. For transaction types where money is sent from the account, it warns when the transaction amount exceeds the sender's recorded balance.

## 📊 Evaluation

Because financial fraud datasets are highly imbalanced, accuracy alone is not sufficient to evaluate the model.

This project therefore considers:

* **Precision** — How many predicted fraud transactions were actually fraudulent
* **Recall** — How many actual fraud transactions were detected
* **F1-score** — Balance between precision and recall
* **Confusion Matrix** — Breakdown of correct and incorrect predictions

## ⚠️ Disclaimer

This is an educational machine learning project and should not be considered a production-ready banking fraud detection system.

The model uses only three transaction features and is intended to demonstrate the machine learning workflow rather than provide real-world financial security.

## 👨‍💻 Author

**Taher Tamboli**

Computer Science Engineering Student

GitHub: [@tahirtamboli0149-gif](https://github.com/tahirtamboli0149-gif)
