from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# 1. Load FAQ documents
loader = TextLoader("./docs/faq.txt", encoding="utf-8")
docs = loader.load()

# 2. Split text into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(docs)

# 3. Generate embeddings & build the DB
embeddings = OllamaEmbeddings(model="mistral")
db = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")
db.persist()

print("✅ Vector database created successfully.")
