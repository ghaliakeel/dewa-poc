#!/usr/bin/env python3
import argparse
import os
import glob
from typing import List
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import numpy as np
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    PyPDFLoader,

)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.docstore.document import Document
import numpy as np

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

from constants import CHROMA_SETTINGS
import chromadb

# Load environment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
chunk_size = 1000
chunk_overlap = 100
embeddings = GPT4AllEmbeddings()
csv_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
# Custom document loaders
# class MyElmLoader(UnstructuredEmailLoader):
#     """Wrapper to fallback to text/plain when default does not work"""

#     def load(self) -> List[Document]:
#         """Wrapper adding fallback for elm without html"""
#         try:
#             try:
#                 doc = UnstructuredEmailLoader.load(self)
#             except ValueError as e:
#                 if 'text/html content not found in email' in str(e):
#                     # Try plain text
#                     self.unstructured_kwargs["content_source"]="text/plain"
#                     doc = UnstructuredEmailLoader.load(self)
#                 else:
#                     raise
#         except Exception as e:
#             # Add file_path to exception message
#             raise type(e)(f"{self.file_path}: {e}") from e

#         return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    # ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    # ".pdf": (PyMuPDFLoader, {}),
    ".pdf": (PyPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(request):
    if 'file' not in request.files:
        return None, {"error": 'no file'}, 400
    files = request.files.getlist('file')  # img.open
    try:
        if files[0].filename == '':
            return None,  {"error": 'no file selected'}, 400
    except:
        if files.filename == '':
            return None,  {"error": 'no file selected'}, 400
    results = []
    results_csv = []
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    if not os.path.exists('csv'):
        os.makedirs('csv')

    for file in files:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower()

        if ext in LOADER_MAPPING:
            loader_class, loader_args = LOADER_MAPPING[ext]
            if ext.lower() !='.csv':
                file.save(f"tmp/{file.filename}")
                loader = loader_class(f"tmp/{file.filename}", **loader_args)
                results.extend(loader.load())
                os.remove(f'tmp/{file.filename}')
            else:
                file.save(f"csv/{file.filename}")

        else:
            return None, None, {"error": f"Unsupported file extension '{ext}'"}, 400
    return results, results_csv, {"message": "file uploaded successfully"}, 200


def process_documents(request):

    print(f"Loading documents from {source_directory}")
    documents, documents_csv, msg, code = load_single_document(request)

    if not documents and not documents_csv:
        return None, None, {"error": "file was not uploaded"}, 400
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    if documents_csv:
        csv_chunks = documents_csv
    else:
        csv_chunks = None
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts,csv_chunks, msg, code


def does_vectorstore_exist(persist_directory, embeddings, CHROMA_SETTINGS, chroma_client) -> bool:
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings,client_settings=CHROMA_SETTINGS, client=chroma_client)
    if not db.get()['documents']:
        return False
    return True


def store_into_vector(text,csv_chunks):
    # Create embeddings

    # Chroma client
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)

    if does_vectorstore_exist(persist_directory, embeddings,CHROMA_SETTINGS, chroma_client):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        if text:
            # print(text)
            db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
            # print(f"Creating embeddings. May take some minutes...")
            db.aadd_documents(text)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = text
        print(f"Creating embeddings. May take some minutes...") # do this here also
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS, client=chroma_client)
    db.persist()
    db = None

    return {"message": "file uploaded successfully"}, 200


def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()