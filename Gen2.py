import streamlit as st
import pandas as pd
import pycaret.classification as clf
import pycaret.regression as reg
import smtplib
import ssl
from email.message import EmailMessage
import matplotlib
import warnings
import imaplib
import email
import requests

# === Fix font warnings ===
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
warnings.filterwarnings("ignore", module="matplotlib")

# === Email & API Config ===
EMAIL_ADDRESS = "akashvishnu6680@gmail.com"
EMAIL_PASSWORD = "swpe pwsx ypqo hgnk"  # Use Gmail App Password
TOGETHER_API_KEY = "tgp_v1_ecSsk1__FlO2mB_gAaaP2i-Affa6Dv8OCVngkWzBJUY"

IMAP_SERVER = "imap.gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465

# === Streamlit Config ===
st.set_page_config(page_title="🤖 Agentic AI ML & Email App", layout="wide")
st.title("🤖 Agentic AI: ML + Email Auto-Responder")

# === EMAIL FUNCTIONS ===
def fetch_latest_email():
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        mail.select("inbox")
        result, data = mail.search(None, 'UNSEEN')
        ids = data[0].split()

        if not ids:
            return None, None, None

        latest_email_id = ids[-1]
        result, msg_data = mail.fetch(latest_email_id, "(RFC822)")
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
        st.error(f"❌ Email error: {e}")
        return None, None, None

def generate_reply_together_ai(msg):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [
            {"role": "system", "content": "You're a helpful assistant replying to business emails clearly."},
            {"role": "user", "content": msg}
        ],
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

def send_reply(to_email, subject, body):
    try:
        msg = EmailMessage()
        msg["Subject"] = "RE: " + subject
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = to_email
        msg.set_content(body)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

        return True
    except Exception as e:
        st.error(f"❌ Send error: {e}")
        return False

# === MACHINE LEARNING SECTION ===
st.sidebar.header("📊 ML AutoML Setup")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
ml_type = st.sidebar.selectbox("Select task type", ["Classification", "Regression"])
target = st.sidebar.text_input("Enter your target column")

if uploaded_file and target:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Dataset Preview")
    st.dataframe(df.head())

    if ml_type == "Classification":
        clf.setup(data=df, target=target, session_id=123, fold=3, n_jobs=-1, html=False)
        model = clf.compare_models()
        tuned_model = clf.tune_model(model)
        clf.evaluate_model(tuned_model)
        clf.interpret_model(tuned_model)
        clf.save_model(tuned_model, 'my_model')
        st.success("✅ Classification model trained and saved.")
    else:
        reg.setup(data=df, target=target, session_id=123, fold=3, n_jobs=-1, html=False)
        model = reg.compare_models()
        tuned_model = reg.tune_model(model)
        reg.evaluate_model(tuned_model)
        reg.interpret_model(tuned_model)
        reg.save_model(tuned_model, 'my_model')
        st.success("✅ Regression model trained and saved.")

# === EMAIL UI & REPLY ===
st.markdown("---")
st.header("📧 Email Auto-Responder")

if st.button("📥 Check Email and Auto-Reply"):
    from_email, subject, message = fetch_latest_email()

    if from_email:
        st.subheader("📨 New Email")
        st.markdown(f"*From:* {from_email}")
        st.markdown(f"*Subject:* {subject}")
        st.text_area("Client Message", value=message, height=150)

        ai_reply = generate_reply_together_ai(message)
        st.text_area("🤖 AI Generated Reply", value=ai_reply, height=180)

        if send_reply(from_email, subject, ai_reply):
            st.success("✅ Reply sent to client.")
            st.balloons()
    else:
        st.info("📭 No new emails found.")
