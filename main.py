from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os

# Load documents
loader = TextLoader("faq.txt")
documents = loader.load()

# Split documents
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Use OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Create vectorstore
db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever()

# LLM and QA chain
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Query handler
def ask_question(question):
    return qa.run(question)
