import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

load_dotenv()

embedding = OpenAIEmbeddings()
db = Chroma(persist_directory="vectordb", embedding_function=embedding)
retriever = db.as_retriever()

llm = ChatOpenAI(model_name="gpt-3.5-turbo")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def ask_question(question):
    return qa.run(question)

def load_and_split(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)

def save_to_vectorstore(docs):
    Chroma.from_documents(documents=docs, embedding=embedding, persist_directory="vectordb")
