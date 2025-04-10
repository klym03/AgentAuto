from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, query):
        return self.model.encode(query).tolist()


def create_vector_store(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_text(text)

    embedding_model = SentenceTransformerEmbeddings('all-MiniLM-L6-v2')  # Обгортка для SentenceTransformer

    vector_store = FAISS.from_texts(documents, embedding_model)

    return vector_store
