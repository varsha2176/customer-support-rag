from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from textblob import TextBlob

embedding = OllamaEmbeddings(model="mistral")
db = Chroma(persist_directory="vectordb", embedding_function=embedding)
retriever = db.as_retriever()
llm = Ollama(model="mistral")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def ask_question(question):
    return qa.run(question)

def get_sentiment(text):
    text_lower = text.lower()

    # Manual positive phrases
    positive_keywords = ["thank you", "thanks", "awesome", "great", "good", "appreciate", "well done", "cool"]
    negative_keywords = ["not working", "problem", "issue", "error", "bad", "worst", "broken", "failed"]

    if any(word in text_lower for word in negative_keywords):
        return "negative"
    if any(word in text_lower for word in positive_keywords):
        return "positive"

    # Fallback to TextBlob
    polarity = TextBlob(text).sentiment.polarity
    if polarity < -0.2:
        return "negative"
    elif polarity > 0.2:
        return "positive"
    else:
        return "neutral"


def get_intent(text):
    text = text.lower()
    if "refund" in text or "return" in text:
        return "refund_request"
    elif "problem" in text or "not working" in text:
        return "technical_issue"
    elif "thank" in text:
        return "gratitude"
    else:
        return "general"
