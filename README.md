# Spam Message Classifier

An NLP text-classification application that labels a message as **spam** or **ham (legitimate)** using TF-IDF features and Multinomial Naive Bayes.

## Features

- Downloads the SMS Spam Collection dataset from the UCI Machine Learning Repository.
- Cleans and deduplicates messages.
- Stratified train/test split with a fixed random seed.
- TF-IDF text features with unigrams and bigrams.
- Multinomial Naive Bayes classifier.
- Accuracy, spam precision, spam recall, F1-score, classification report, and confusion matrix.
- Streamlit interface for testing custom messages and viewing class probabilities.

## Tech stack

Python, Pandas, Scikit-learn, TF-IDF, Multinomial Naive Bayes, Joblib, Streamlit.

## Setup on Windows / VS Code

Open this folder in VS Code. Open Terminal → New Terminal and run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Train the classifier (requires internet for the first dataset download):

```powershell
python train.py
```

Run the application:

```powershell
streamlit run app.py
```

Open `http://localhost:8501` if the browser does not launch automatically.

## Dataset

**SMS Spam Collection** from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The training script downloads the official ZIP archive automatically and extracts `SMSSpamCollection` into `data/`. The dataset contains labeled SMS examples (`ham` and `spam`). Do not redistribute the dataset separately without checking its terms.

## Project structure

```text
02_Spam_Message_Classifier/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── README.txt
└── models/
    └── .gitkeep
```

After training, `data/SMSSpamCollection`, `models/spam_classifier.joblib`, and `models/metrics.json` are generated.

## Evaluation definitions

- **Precision (spam):** Of messages flagged as spam, how many were actually spam?
- **Recall (spam):** Of all actual spam messages, how many did the model catch?
- **F1-score (spam):** Harmonic mean of spam precision and spam recall.
- **Confusion matrix:** Counts correct and incorrect predictions for each class.

## Limitations

- This is a learning classifier, not a production email security gateway.
- It is trained on SMS data; long emails, attachments, images, and phishing URLs are outside its scope.
- A false negative can let spam through; a false positive can flag a legitimate message.
- Model probabilities are not calibrated guarantees. Do not use the output as the sole security decision.

## GitHub publishing

1. Create a public repository named `spam-message-classifier-nlp`.
2. In this folder, run:

```powershell
git init
git add .
git commit -m "Build NLP spam message classifier"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/spam-message-classifier-nlp.git
git push -u origin main
```

3. Replace `YOUR-USERNAME` with your GitHub username.
4. Add a screenshot of the running app and the actual metrics from `models/metrics.json`.
5. Do not upload `.venv/`, cache files, or the downloaded dataset. The training script downloads the dataset again when needed.
