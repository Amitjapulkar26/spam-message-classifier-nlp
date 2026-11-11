# 📩 Spam Message Classifier using NLP

An NLP-based machine learning application that classifies SMS messages as **Spam** or **Ham (Legitimate)** using TF-IDF feature extraction and a Multinomial Naive Bayes classifier.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?logo=scikitlearn" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/NLP-Text%20Classification-purple" alt="NLP">
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Status-Deployed-brightgreen" alt="Deployed">
</p>

---

## 🚀 Live Demo

<p align="center">
  <a href="https://spam-message-classifier-np-ngni3p3jadvusmntpbappch.streamlit.app/">
    <strong>🌐 OPEN SPAM MESSAGE CLASSIFIER</strong>
  </a>
</p>

Try the deployed application. Enter any SMS or text message and check whether the trained machine learning model identifies it as spam or legitimate.

The application also displays prediction confidence and class probabilities.

## 📸 Project Screenshots

Screenshots of the actual application interface, message classification results, and model evaluation will be added here.

## 🎯 Project Overview

The Spam Message Classifier is a Natural Language Processing (NLP) application designed to identify unwanted or promotional SMS messages.

It uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert text messages into numerical features and a **Multinomial Naive Bayes** classifier to predict whether a message is spam or ham.

The application provides an interactive Streamlit interface where users can enter custom messages and view the model's classification results.

## ✨ Key Features

- Downloads the SMS Spam Collection dataset from the UCI Machine Learning Repository.
- Cleans and deduplicates messages.
- Stratified train/test split with a fixed random seed.
- TF-IDF text features with unigrams and bigrams.
- Multinomial Naive Bayes classifier.
- Accuracy, spam precision, spam recall, and F1-score evaluation.
- Classification report and confusion matrix.
- Streamlit interface for testing custom messages.
- Displays spam/ham prediction and class probabilities.
- Deployed application accessible through a public web link.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Core programming language |
| Pandas | Data handling and preprocessing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and model evaluation |
| TF-IDF | Text feature extraction |
| Multinomial Naive Bayes | Message classification |
| Joblib | Model serialization and loading |
| Streamlit | Interactive web application |
| Matplotlib | Data visualization |
| Seaborn | Confusion matrix visualization |

## 🧠 Machine Learning Workflow

### 1. Dataset Collection

The SMS Spam Collection dataset is obtained from the UCI Machine Learning Repository.

### 2. Data Preprocessing

- Load the SMS dataset.
- Clean and deduplicate messages.
- Separate message text and labels.
- Prepare data for model training.

### 3. Feature Extraction

TF-IDF converts text messages into numerical feature vectors.

The classifier uses:
- Unigrams (single words).
- Bigrams (two-word combinations).
- English stop-word removal.
- Lowercase text processing.

### 4. Model Training

A Multinomial Naive Bayes classifier is trained using the extracted TF-IDF features.

### 5. Model Evaluation

The model is evaluated using:
- Accuracy
- Spam precision
- Spam recall
- Spam F1-score
- Classification report
- Confusion matrix

### 6. Prediction

Users enter a custom message in the Streamlit interface. The trained model predicts whether the message is spam or legitimate.

### Workflow Diagram

```text
       SMS Dataset
            |
            v
     Data Cleaning
            |
            v
    Train/Test Split
            |
            v
    TF-IDF Extraction
            |
            v
 Multinomial Naive Bayes
            |
            v
     Model Evaluation
            |
            v
    Trained Model Saved
            |
            v
    Streamlit Interface
            |
            v
   Spam / Ham Prediction
```

## 📊 Model Performance

The trained classifier achieved the following results on the held-out test dataset.

<details>
<summary><strong>View Model Evaluation Results</strong></summary>

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 98.16% |
| Spam Precision | 100.00% |
| Spam Recall | 85.16% |
| Spam F1-score | 91.98% |
| Training Samples | 4,126 |
| Testing Samples | 1,032 |

</details>

### Understanding the Results

- **Accuracy – 98.16%:** The percentage of correctly classified messages in the test dataset.
- **Spam Precision – 100%:** Of the messages predicted as spam, all were actually spam in this test set.
- **Spam Recall – 85.16%:** The model identified approximately 85% of the actual spam messages.
- **Spam F1-score – 91.98%:** The balance between spam precision and recall.

**Important:** The model does not identify every spam message. Its 85.16% spam recall means some actual spam messages may be missed.

## 📚 Dataset

**SMS Spam Collection Dataset** from the UCI Machine Learning Repository.

Official dataset:

[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

The dataset contains labeled SMS messages belonging to two categories:

- **Ham:** Legitimate messages.
- **Spam:** Unwanted or promotional messages.

The training script downloads the official ZIP archive automatically and extracts `SMSSpamCollection` into the `data/` directory.

The dataset should not be redistributed separately without checking its applicable terms.

## ⚙️ Installation and Setup on Windows / VS Code

### 1. Clone the repository

```powershell
git clone https://github.com/Amitjapulkar26/spam-message-classifier-nlp.git
```

### 2. Open the project folder

```powershell
cd spam-message-classifier-nlp
```

Open this folder in VS Code.

Go to **Terminal → New Terminal**.

### 3. Create a Python 3.11 virtual environment

```powershell
py -3.11 -m venv .venv
```

### 4. Activate the environment

```powershell
.venv\Scripts\activate
```

### 5. Upgrade pip and install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Train the classifier

The first training run requires internet access to download the dataset.

```powershell
python train.py
```

After training, the following files are generated:

- `models/spam_classifier.joblib`
- `models/metrics.json`
- `data/SMSSpamCollection`

### 7. Run the Streamlit application

```powershell
streamlit run app.py
```

A browser tab should open automatically.

If it does not, visit:

`http://localhost:8501`

## 📂 Project Structure

```text
02_Spam_Message_Classifier/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── README.txt
│
└── models/
    ├── .gitkeep
    ├── spam_classifier.joblib
    └── metrics.json
```

After training, the dataset and model artifacts are generated automatically.

The trained model is saved as `models/spam_classifier.joblib`, while evaluation results are stored in `models/metrics.json`.

## 📖 Evaluation Definitions

- **Precision (Spam):** Of the messages flagged as spam, how many were actually spam?
- **Recall (Spam):** Of all actual spam messages, how many did the model catch?
- **F1-score (Spam):** Harmonic mean of spam precision and spam recall.
- **Confusion Matrix:** Counts correct and incorrect predictions for each class.
- **Accuracy:** Percentage of all test messages classified correctly.

## 🎥 Project Demonstration

For an internship or technical evaluation, demonstrate the following:

1. Open the deployed Streamlit application.
2. Enter a legitimate SMS message.
3. Click the classification button.
4. Show the predicted class and confidence.
5. Enter a spam message and classify it.
6. Explain TF-IDF and Multinomial Naive Bayes.
7. Show the model evaluation results.

## 🗣️ Project Explanation for Evaluation

My second project is a Spam Message Classifier developed using Python, Natural Language Processing, and Machine Learning.

I used the SMS Spam Collection dataset from the UCI Machine Learning Repository. First, I cleaned and prepared the messages. Then I applied TF-IDF to convert text into numerical features and trained a Multinomial Naive Bayes classifier to distinguish between spam and legitimate messages.

The model achieved 98.16% test accuracy, with 100% spam precision, 85.16% spam recall, and a 91.98% spam F1-score.

I developed an interactive Streamlit application where users can enter custom messages and view the predicted category and class probabilities.

This project helped me gain practical experience in NLP, text preprocessing, feature extraction, machine learning model training, evaluation, and cloud deployment.

## ⚠️ Limitations

- This is a learning classifier, not a production email security gateway.
- It is trained on SMS data; long emails, attachments, images, and phishing URLs are outside its scope.
- A false negative can let spam through.
- A false positive can incorrectly flag a legitimate message.
- Model probabilities are not calibrated guarantees.
- Do not use the output as the sole security decision.

## 🔮 Future Improvements

- Improve spam recall while maintaining high precision.
- Test additional machine learning algorithms.
- Experiment with advanced NLP techniques.
- Add support for email text classification.
- Improve model evaluation visualizations.
- Explore transformer-based text classification models.

## 👨‍💻 Developer

**Amit Japulkar**

AI & Machine Learning Engineering Student

- GitHub: [@Amitjapulkar26](https://github.com/Amitjapulkar26)
- LinkedIn: [Connect on LinkedIn](https://www.linkedin.com/in/amit-japulkar-939371393/)

---

<p align="center">
  <strong>Built with Python, NLP, Scikit-learn, and Streamlit.</strong>
  <br>
  If you find this project useful, consider giving the repository a ⭐.
</p>
