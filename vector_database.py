# vector_database.py
"""
Vector database utilities for the RAG pipeline.
- Builds a FAISS index from PDFs in `pdfs/`
- Exposes helpers: build_faiss_index, load_faiss_db, add_pdf_and_rebuild, similarity_search
- Keeps an in-memory `faiss_db` handle for quick access
"""

import os
from typing import List, Tuple, Optional

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Paths (customize if you want)
DATA_PATH = "pdfs/"
DB_FAISS_PATH = "vectorstore/faiss_index"

# Global in-memory handle
faiss_db: Optional[FAISS] = None


def ensure_dirs():
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)


def load_pdf_files(data_dir: str = DATA_PATH) -> List[Document]:
    """
    Load all PDFs from `data_dir` using LangChain's DirectoryLoader + PyPDFLoader.
    Returns a list of Document objects.
    """
    loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


def create_chunks(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    Split documents into text chunks suitable for embedding and indexing.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    return chunks


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Returns a HuggingFaceEmbeddings instance. Change model_name if you prefer a different embedding model.
    """
    return HuggingFaceEmbeddings(model_name=model_name)


def build_faiss_index(data_path: str = DATA_PATH, save_path: str = DB_FAISS_PATH, 
                      chunk_size: int = 500, chunk_overlap: int = 50,
                      embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    """
    Read PDFs from data_path, chunk them, compute embeddings, and build a FAISS index.
    The index is saved to `save_path` and the in-memory `faiss_db` is updated.
    Returns the FAISS object.
    """
    ensure_dirs()
    documents = load_pdf_files(data_path)
    if not documents:
        raise RuntimeError(f"No PDF documents found in {data_path}. Place PDFs there or upload via the UI.")

    text_chunks = create_chunks(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embedding_model = get_embedding_model(embedding_model_name)
    db = FAISS.from_documents(text_chunks, embedding_model)
    # ensure destination directory exists
    os.makedirs(save_path, exist_ok=True)
    db.save_local(save_path)
    global faiss_db
    faiss_db = db
    return db


# vector_database.py update (replace the load_faiss_db implementation)
def load_faiss_db(save_path: str = DB_FAISS_PATH, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    """
    Load a previously saved FAISS index into memory and return it.

    Note: Some LangChain versions require `allow_dangerous_deserialization=True` to allow
    pickle-based index metadata to be deserialized. Only set that if you trust the saved files.
    """
    global faiss_db
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"FAISS DB not found at {save_path}. Build it first with build_faiss_index().")

    embedding_model = get_embedding_model(embedding_model_name)

    # Try to load with explicit allow_dangerous_deserialization flag (newer LangChain)
    try:
        faiss_db = FAISS.load_local(save_path, embedding_model, allow_dangerous_deserialization=True)
        return faiss_db
    except TypeError:
        # Older LangChain versions may not accept that kwarg; fall back to the simple call
        faiss_db = FAISS.load_local(save_path, embedding_model)
        return faiss_db
    except Exception as exc:
        # Surface the original error so Streamlit can show it
        raise



def add_pdf_and_rebuild(uploaded_file, data_path: str = DATA_PATH, save_path: str = DB_FAISS_PATH,
                        chunk_size: int = 500, chunk_overlap: int = 50,
                        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    """
    Save an uploaded Streamlit file-like object into `data_path` and rebuild the FAISS index.
    `uploaded_file` should be a Streamlit UploadedFile-like object (has .name and .getbuffer()).
    """
    ensure_dirs()
    dest_path = os.path.join(data_path, uploaded_file.name)
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return build_faiss_index(data_path=data_path, save_path=save_path,
                             chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                             embedding_model_name=embedding_model_name)


def similarity_search(query: str, k: int = 4) -> List[Tuple[Document, float]]:
    """
    Convenience wrapper around FAISS similarity_search_with_score.
    Returns list of (Document, score) tuples.
    """
    if faiss_db is None:
        raise RuntimeError("FAISS DB is not loaded. Call build_faiss_index() or load_faiss_db() first.")
    results = faiss_db.similarity_search_with_score(query, k=k)
    return results


# If run as script, provide a small CLI to build or load the index
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build or load FAISS index for RAG")
    parser.add_argument("--build", action="store_true", help="Build FAISS index from pdfs/")
    parser.add_argument("--load", action="store_true", help="Load FAISS index from disk")
    args = parser.parse_args()

    if args.build:
        print("Building FAISS index from PDFs in pdfs/ ...")
        build_faiss_index()
        print("Done. Index saved to", DB_FAISS_PATH)
    if args.load:
        print("Loading FAISS index from disk ...")
        load_faiss_db()
        print("Done. Index loaded into memory.")