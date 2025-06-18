import streamlit as st
import os
from main import ask_question, get_sentiment, get_intent

st.set_page_config(page_title="Customer Support RAG", page_icon="🤖", layout="centered")
st.title("📞 Customer Support Chatbot (RAG + Ollama)")
st.markdown("Upload an FAQ or ask your question directly below.")

# File upload section
uploaded = st.file_uploader("📁 Upload FAQ file (.txt)", type=["txt"])
if uploaded:
    with open("./docs/faq.txt", "wb") as f:
        f.write(uploaded.read())
    st.success("✅ FAQ file saved.")
    if st.button("🔄 Rebuild Database"):
        os.system("python build_index.py")
        st.success("✅ Knowledge base rebuilt. You can ask questions now.")

# Chat interaction
st.markdown("---")
query = st.text_input("🗣️ Enter your question:")
if query:
    sentiment = get_sentiment(query)
    intent = get_intent(query)
    response = ""
    with st.spinner("Thinking..."):
        response = ask_question(query)

    st.markdown(f"**Sentiment:** `{sentiment}` | **Intent:** `{intent}`")
    if sentiment == "negative" or intent == "technical_issue":
        st.warning("⚠️ This seems serious. Escalate to human support: support@example.com")

    st.success("💬 Response:")
    st.write(response)
