import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import smtplib
import seaborn as sns
import matplotlib.pyplot as plt
import os
from email.message import EmailMessage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from imblearn.over_sampling import SMOTE
from matplotlib.backends.backend_pdf import PdfPages
import ssl
import imaplib
import email

# === Email Config ===
EMAIL_ADDRESS = st.secrets["EMAIL_ADDRESS"]
EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
IMAP_SERVER = "imap.gmail.com"

# === Streamlit Config ===
st.set_page_config(page_title="⚡ Fast AutoML Agent", layout="wide")
st.title("🤖 Fast AutoML + Email Agent")

# === Upload and Detect Task ===
st.sidebar.header("📊 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a dataset", type=["csv", "xlsx"])

if uploaded_file:
    ext = uploaded_file.name.split(".")[-1]
    df = pd.read_csv(uploaded_file) if ext == "csv" else pd.read_excel(uploaded_file)
    st.write("✅ Data Preview")
    st.dataframe(df.head())

    target = st.sidebar.selectbox("Select Target Column", df.columns)

    # Task detection
    task_type = "classification" if df[target].nunique() <= 20 or df[target].dtype == 'object' else "regression"
    st.info(f"🔍 Detected Task Type: **{task_type.title()}**")

    X = df.drop(columns=[target])
    y = df[target]

    # Handle imbalance (optional for classification)
    if task_type == "classification":
        X = pd.get_dummies(X)
        sm = SMOTE()
        X, y = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === Model list ===
    classifiers = [
        ("RandomForest", RandomForestClassifier()),
        ("GradientBoosting", GradientBoostingClassifier()),
        ("KNN", KNeighborsClassifier()),
        ("SVM", SVC(probability=True)),
        ("LogisticRegression", LogisticRegression(max_iter=1000))
    ]

    regressors = [
        ("RandomForest", RandomForestRegressor()),
        ("GradientBoosting", GradientBoostingRegressor()),
        ("KNN", KNeighborsRegressor()),
        ("SVM", SVR()),
        ("LinearRegression", LinearRegression())
    ]

    models = classifiers if task_type == "classification" else regressors
    scores = []
    best_models = []

    for name, model in models:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = accuracy_score(y_test, preds) if task_type == "classification" else r2_score(y_test, preds)
        scores.append((name, score))
        best_models.append((name, model))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_models = scores[:3]

    st.subheader("🏆 Top 3 Models")
    for name, score in top_models:
        st.markdown(f"- **{name}** → Score: `{score:.2f}`")

    # === Generate PDF Report ===
    with PdfPages("model_report.pdf") as pdf:
        plt.figure(figsize=(8, 4))
        names, vals = zip(*scores)
        sns.barplot(x=vals, y=names)
        plt.title("Model Comparison")
        plt.xlabel("Accuracy" if task_type == "classification" else "R2 Score")
        pdf.savefig()
        plt.close()

    st.success("📄 Model Report generated as PDF.")

    # === Email Report ===
    recipient_email = st.text_input("Enter Client Email to Send Report")
    if st.button("📤 Send Report via Email") and recipient_email:
        msg = EmailMessage()
        msg["Subject"] = f"AutoML Report - {task_type.title()}"
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = recipient_email
        msg.set_content("Attached is the AutoML model performance report.\n\nRegards,\nAgentic AI")

        with open("model_report.pdf", "rb") as f:
            msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename="model_report.pdf")

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

        st.success(f"✅ Report sent to {recipient_email}")

# === Email Auto-Responder ===
st.markdown("---")
st.header("📬 Auto Email Response (From Gmail Inbox)")

def fetch_latest_email():
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        mail.select("inbox")
        result, data = mail.search(None, 'UNSEEN')
        ids = data[0].split()

        if not ids:
            return None, None, None

        latest_id = ids[-1]
        result, msg_data = mail.fetch(latest_id, "(RFC822)")
        raw_email = msg_data[0][1]
        email_message = email.message_from_bytes(raw_email)

        from_email = email_message["From"]
        subject = email_message["Subject"]
        body = ""

        if email_message.is_multipart():
            for part in email_message.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode()
                    break
        else:
            body = email_message.get_payload(decode=True).decode()

        return from_email, subject, body

    except Exception as e:
        st.error(f"❌ Error fetching email: {e}")
        return None, None, None

def send_auto_reply(to_email, subject, reply_content):
    try:
        msg = EmailMessage()
        msg["Subject"] = f"RE: {subject}"
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = to_email
        msg.set_content(reply_content)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

        return True
    except Exception as e:
        st.error(f"❌ Failed to send reply: {e}")
        return False

def generate_ai_reply(message):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant replying to business emails."},
            {"role": "user", "content": message}
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

if st.button("📥 Check & Auto-Reply to Latest Email"):
    from_email, subject, body = fetch_latest_email()
    if from_email:
        st.subheader("📨 Incoming Email")
        st.markdown(f"**From:** {from_email}")
        st.markdown(f"**Subject:** {subject}")
        st.text_area("Message", value=body, height=150)

        reply = generate_ai_reply(body)
        st.text_area("🤖 AI Reply", value=reply, height=180)

        if send_auto_reply(from_email, subject, reply):
            st.success("✅ Reply sent successfully.")
    else:
        st.info("📭 No new unread emails.")
