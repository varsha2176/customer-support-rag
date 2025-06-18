import streamlit as st
from main import ask_question

st.set_page_config(page_title="Customer Support Chatbot")
st.title("Customer Support FAQ Bot 🤖")
st.markdown("Ask me anything based on the company FAQs!")

query = st.text_input("Enter your question here")

if query:
    response = ask_question(query)
    st.write("### Answer:")
    st.write(response)
