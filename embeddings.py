from sentence_transformers import SentenceTransformer
import os
import pickle
from PyPDF2 import PdfReader

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MODEL_PATH = "embeddings.pkl"
DATA_PATH = "data"
MAX_CHUNK_SIZE = 100  # Max tokens (words or characters) per chunk


def chunk_text(text, max_chunk_size=MAX_CHUNK_SIZE):
    """
    Splits the text into smaller chunks, each with a max length of `max_chunk_size`.
    """
    words = text.split()
    chunks = [words[i:i + max_chunk_size] for i in range(0, len(words), max_chunk_size)]
    return [" ".join(chunk) for chunk in chunks]


def create_embeddings():
    """
    Reads PDFs, creates embeddings for each chunk, and stores them in a file.
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    documents = []
    chunked_documents = []

    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".pdf"):
            pdf_reader = PdfReader(os.path.join(DATA_PATH, filename))
            text = "".join([page.extract_text() for page in pdf_reader.pages])

            # Chunk the document text
            chunks = chunk_text(text)
            chunked_documents.extend(chunks)

    # Create embeddings for each chunk
    embeddings = model.encode(chunked_documents)

    # Save the documents and embeddings
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"documents": chunked_documents, "embeddings": embeddings}, f)

    print("Embeddings created and saved!")


def load_embeddings():
    """
    Loads precomputed embeddings from a file.
    """
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)
