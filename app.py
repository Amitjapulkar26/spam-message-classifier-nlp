
"""Professional SMS Spam Message Classifier."""

from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

# ==================================================
# 1. CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "spam_classifier.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(
    page_title="Spam Classifier | NLP Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================================================
# 2. CUSTOM CSS
# ==================================================

st.markdown("""
<style>

/* Application background */
.stApp {
    background-color: #0e1117;
}

/* Main container */
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Hero section */
.hero {
    padding: 30px;
    border-radius: 18px;
    background: linear-gradient(120deg, #102a43, #087e8b);
    margin-bottom: 30px;
    border: 1px solid #245b70;
}

.hero h1 {
    color: #ffffff !important;
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #d9f4f3 !important;
    font-size: 1rem;
    margin-bottom: 0;
}

/* Result card */
.result {
    padding: 24px;
    border-radius: 15px;
    margin-top: 15px;
    margin-bottom: 20px;
}

.result-title {
    font-size: 1.5rem;
    font-weight: 800;
    margin-bottom: 12px;
}

.result-confidence {
    color: #374151 !important;
    font-size: 1.05rem;
    font-weight: 600;
}

.result-note {
    color: #4b5563 !important;
    font-size: 0.9rem;
    margin-top: 10px;
}

/* Headings */
h1, h2, h3 {
    color: #f1f5f9 !important;
}

/* Text */
p, label {
    color: #e2e8f0;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #1c2533;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #334155;
}

[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* Text area */
.stTextArea textarea {
    border-radius: 10px;
    border: 1px solid #475569;
}

/* Buttons */
.stButton button {
    border-radius: 9px;
    font-weight: 700;
    min-height: 45px;
}

/* Divider */
hr {
    border-color: #334155;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# 3. HEADER
# ==================================================

st.markdown("""
<div class="hero">
    <h1>🛡️ Spam Message Classifier</h1>
    <p>
        Natural Language Processing
        • TF-IDF
        • Multinomial Naive Bayes
    </p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# 4. LOAD MODEL
# ==================================================

@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Trained model not found!")

    st.code(
        "python train.py\npython -m streamlit run app.py",
        language="powershell"
    )

    st.stop()

try:
    pipeline = load_pipeline()

except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()

# ==================================================
# 5. MESSAGE INPUT
# ==================================================

st.subheader("🔍 Test a Message")

st.write(
    "Enter an SMS or email message to check whether "
    "it is spam or legitimate."
)

examples = {
    "Choose an example": "",

    "Legitimate message":
        "Hi, are we still meeting at the library at 5 pm today?",

    "Promotional spam":
        "Congratulations! You have won a free prize. "
        "Claim your cash reward now by clicking this link!",

    "Banking spam":
        "URGENT! Your bank account will be blocked. "
        "Verify your details immediately to receive your reward.",

    "Personal message":
        "Hey, are you coming to college tomorrow? "
        "Let's meet after class.",

    "Prize scam":
        "URGENT! You have won a FREE cash prize of $1000. "
        "Call now to claim your reward!"
}

selected = st.selectbox(
    "Load a sample message",
    list(examples.keys())
)

message = st.text_area(
    "Message text",
    value=examples[selected],
    height=140,
    placeholder="Paste your SMS or email message here..."
)

# ==================================================
# 6. CLASSIFICATION
# ==================================================

if st.button(
    "🚀 Classify Message",
    type="primary",
    use_container_width=True
):

    if not message.strip():

        st.warning("Please enter a message before classification.")

    else:

        try:

            # Model prediction
            prediction = pipeline.predict([message])[0]

            probabilities = pipeline.predict_proba([message])[0]

            classes = list(pipeline.classes_)

            scores = {
                label: float(probabilities[index])
                for index, label in enumerate(classes)
            }

            confidence = scores[prediction]

            is_spam = prediction == "spam"

            # Result styling
            if is_spam:

                title = "🚨 SPAM DETECTED"
                color = "#b42318"
                background = "#fef2f2"
                border = "#fca5a5"

                description = (
                    "This message has been classified as potentially "
                    "unwanted or fraudulent."
                )

            else:

                title = "✅ LIKELY LEGITIMATE (HAM)"
                color = "#067647"
                background = "#ecfdf3"
                border = "#86efac"

                description = (
                    "This message has been classified as legitimate "
                    "based on the trained model."
                )

            # Display prediction card
            # IMPORTANT: HTML must start at column zero.
            st.markdown(
                f"""
<div class="result"
     style="background:{background};border:1px solid {border};">
    <div class="result-title" style="color:{color};">
        {title}
    </div>
    <div class="result-confidence">
        Model Confidence:
        <strong style="color:{color};">{confidence:.2%}</strong>
    </div>
    <div class="result-note">
        {description}
    </div>
</div>
""",
                unsafe_allow_html=True
            )

            # ==================================================
            # 7. PROBABILITY CHART
            # ==================================================

            st.subheader("📊 Class Probabilities")

            probability_data = pd.DataFrame(
                {
                    "Probability": [
                        scores.get("ham", 0.0),
                        scores.get("spam", 0.0)
                    ]
                },
                index=["HAM", "SPAM"]
            )

            # Compatible with installed Streamlit versions
            st.bar_chart(
                probability_data,
                height=300
            )

            # Individual probability metrics
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "HAM Probability",
                    f"{scores.get('ham', 0.0):.2%}"
                )

            with col2:
                st.metric(
                    "SPAM Probability",
                    f"{scores.get('spam', 0.0):.2%}"
                )

            st.caption(
                "Model confidence is a probability estimate, "
                "not proof that a message is safe. "
                "Never trust suspicious links based only on this result."
            )

        except Exception as error:

            st.error(f"Classification failed: {error}")

# ==================================================
# 8. MODEL EVALUATION
# ==================================================

st.divider()

st.subheader("📈 Model Evaluation")

if METRICS_PATH.exists():

    try:

        metrics = json.loads(
            METRICS_PATH.read_text(encoding="utf-8")
        )

        # Evaluation metrics
        a, b, c, d = st.columns(4)

        a.metric(
            "Accuracy",
            f"{metrics['accuracy']:.2%}"
        )

        b.metric(
            "Spam Precision",
            f"{metrics['precision_spam']:.2%}"
        )

        c.metric(
            "Spam Recall",
            f"{metrics['recall_spam']:.2%}"
        )

        d.metric(
            "Spam F1-Score",
            f"{metrics['f1_spam']:.2%}"
        )

        st.caption(
            f"Evaluation performed on {metrics['test_samples']:,} "
            f"test messages from {metrics['dataset_samples']:,} "
            f"cleaned unique messages."
        )

        # ==================================================
        # 9. CONFUSION MATRIX
        # ==================================================

        st.subheader("Confusion Matrix")

        cm = metrics["confusion_matrix"]

        confusion_df = pd.DataFrame(
            cm,
            index=["Actual Ham", "Actual Spam"],
            columns=["Predicted Ham", "Predicted Spam"]
        )

        st.dataframe(
            confusion_df,
            use_container_width=True
        )

        # ==================================================
        # 10. CLASSIFICATION REPORT
        # ==================================================

        if "classification_report" in metrics:

            with st.expander("View Detailed Classification Report"):

                report = pd.DataFrame(
                    metrics["classification_report"]
                ).transpose()

                st.dataframe(
                    report,
                    use_container_width=True
                )

    except Exception as error:

        st.error(f"Unable to read evaluation metrics: {error}")

else:

    st.info("Run train.py to generate model evaluation metrics.")

# ==================================================
# 11. FOOTER
# ==================================================

st.divider()

st.markdown("""
<div style="text-align:center; color:#94a3b8; padding:15px;">

    <strong>Spam Message Classifier | NLP Lab</strong>

    <p style="font-size:13px;">
        Developed using Python, Streamlit,
        Scikit-learn and Natural Language Processing.
    </p>

    <p style="font-size:12px;">
        Educational Project • TF-IDF • Multinomial Naive Bayes
    </p>

</div>
""", unsafe_allow_html=True)
