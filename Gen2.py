import streamlit as st
import pandas as pd
import pycaret.classification as clf
import pycaret.regression as reg
import smtplib
import ssl
from email.message import EmailMessage
import matplotlib
import warnings

# === Fix font warnings ===
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
warnings.filterwarnings("ignore", module="matplotlib")

# === Email Config ===
EMAIL_ADDRESS = "akashvishnu6680@gmail.com"
EMAIL_PASSWORD = "swpe pwsx ypqo hgnk"

# === Streamlit UI ===
st.set_page_config(page_title="Smart AutoML App", layout="wide")
st.title("🚀 Smart AutoML with PyCaret")
st.caption("Upload a dataset, select the target, and let AI do the rest!")

# === Upload CSV ===
uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")
    st.dataframe(df.head())

    # === Select target column ===
    target = st.selectbox("🎯 Select the target column", df.columns)

    # === Choose task ===
    task_type = st.radio("🧠 Select the ML task type", ["Classification", "Regression"])

    if st.button("🚀 Run AutoML"):
        st.toast("Running AutoML... Please wait")
        with st.spinner("Training model and tuning..."):

            # Validate target column
            if df[target].isnull().sum() > 0:
                st.error("❌ Target column contains missing values.")
            else:
                try:
                    if task_type == "Classification":
                        clf.setup(data=df, target=target, session_id=123)
                        best_model = clf.compare_models()
                        tuned_model = clf.tune_model(best_model)
                        st.write(clf.pull())
                        clf.evaluate_model(tuned_model)
                        clf.interpret_model(tuned_model)
                        clf.save_model(tuned_model, 'my_model')

                    else:
                        reg.setup(data=df, target=target, session_id=123)
                        best_model = reg.compare_models()
                        tuned_model = reg.tune_model(best_model)
                        st.write(reg.pull())
                        reg.evaluate_model(tuned_model)
                        reg.interpret_model(tuned_model)
                        reg.save_model(tuned_model, 'my_model')

                    st.success("✅ Our model is trained and saved successfully!")
                    st.balloons()

                    # === Email Notification ===
                    def send_email():
                        msg = EmailMessage()
                        msg["Subject"] = "✅ Our ML Model is Ready!"
                        msg["From"] = EMAIL_ADDRESS
                        msg["To"] = EMAIL_ADDRESS
                        msg.set_content(f"""
Hi,

Our machine learning model has been trained successfully using AutoML.

Task Type: {task_type}  
Target Column: {target}

We can now use it to make predictions or deploy it into production.

Kind regards,  
Your AI Assistant 🤖
""")
                        context = ssl.create_default_context()
                        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                            server.send_message(msg)

                    send_email()
                    st.success("📧 Notification email sent: Our ML model is ready!")

                except Exception as e:
                    st.error(f"⚠️ An error occurred: {str(e)}")

# === Divider ===
st.markdown("---")
st.subheader("📨 Client Email Assistant")

# === Client Email AI Reply ===
client_email = st.text_input("📧 Enter client’s email address")
client_question = st.text_area("💬 Type your client's question here")

if st.button("🤖 Generate and Send AI Reply"):
    if client_email and client_question:
        with st.spinner("Agentic AI is replying..."):

            # === Simple reply logic ===
            if "model" in client_question.lower():
                ai_reply = (
                    "Hi,\n\n"
                    "Our machine learning model is fully trained and ready to use. "
                    "You can now upload new data to make predictions or we can help deploy it.\n\n"
                    "Let me know how you'd like to proceed.\n\n"
                    "Best regards,\nYour AI Assistant 🤖"
                )
            elif "how to use" in client_question.lower():
                ai_reply = (
                    "Hi,\n\n"
                    "To use our model, you can upload new data for prediction, or connect it through an API. "
                    "Let us know if you'd like help getting started.\n\n"
                    "Kind regards,\nYour AI Assistant 🤖"
                )
            else:
                ai_reply = (
                    "Hi,\n\n"
                    "Thank you for your message. Could you please clarify what you need help with?\n\n"
                    "Kind regards,\nYour AI Assistant 🤖"
                )

            # Show reply
            st.text_area("✉️ Agentic AI Reply", value=ai_reply, height=150)

            # === Email the client ===
            def send_reply_email():
                msg = EmailMessage()
                msg["Subject"] = "🧠 Response from Your AI Assistant"
                msg["From"] = EMAIL_ADDRESS
                msg["To"] = client_email
                msg.set_content(ai_reply)

                context = ssl.create_default_context()
                with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                    server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                    server.send_message(msg)

            try:
                send_reply_email()
                st.success("📤 Reply sent to client successfully!")
            except Exception as e:
                st.error(f"⚠️ Could not send email: {str(e)}")
    else:
        st.warning("Please fill in both the client email and their message.")
