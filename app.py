import streamlit as st
from main import ask_question, load_and_split, save_to_vectorstore
import os

st.set_page_config(page_title="📞 Customer Support Chatbot (RAG + OpenAI)")
st.title("📞 Customer Support Chatbot (RAG + OpenAI)")
st.markdown("Upload an FAQ or ask your question directly below.")

uploaded_file = st.file_uploader("📁 Upload FAQ file (.txt)", type="txt")

if uploaded_file is not None:
    file_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    with st.spinner("Processing document..."):
        docs = load_and_split(file_path)
        save_to_vectorstore(docs)
    st.success("FAQ uploaded and embedded successfully.")

query = st.text_input("🗣️ Enter your question:")

if query:
    with st.spinner("Getting answer..."):
        try:
            response = ask_question(query)
            st.success(response)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
