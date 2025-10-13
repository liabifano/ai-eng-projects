# Import standard libraries for file handling and text processing
import os, pathlib, textwrap, glob

# Load documents from various sources (URLs, text files, PDFs)
from langchain_community.document_loaders import UnstructuredURLLoader, TextLoader, PyPDFLoader

# Split long texts into smaller, manageable chunks for embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector store to store and retrieve embeddings efficiently using FAISS
from langchain.vectorstores import FAISS

# Generate text embeddings using OpenAI or Hugging Face models
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, SentenceTransformerEmbeddings

# Use local LLMs (e.g., via Ollama) for response generation
from langchain.llms import Ollama

# Build a retrieval chain that combines a retriever, a prompt, and an LLM
from langchain.chains import ConversationalRetrievalChain

# Create prompts for the RAG system
from langchain.prompts import PromptTemplate

print("✅ Libraries imported! You're good to go!")


def ingest_pdfs():
    pdf_paths = glob.glob("data/Everstorm_*.pdf")
    raw_docs = [PyPDFLoader(path).load() for path in pdf_paths]

    return raw_docs

    # print(f"Loaded {len(raw_docs)} PDF pages from {len(pdf_paths)} files.")


def ingest_urls():
    URLS = [
    # --- BigCommerce – shipping & refunds ---
    "https://developer.bigcommerce.com/docs/store-operations/shipping",
    "https://developer.bigcommerce.com/docs/store-operations/orders/refunds",
    # --- Stripe – disputes & chargebacks ---
    # "https://docs.stripe.com/disputes",  
    # --- WooCommerce – REST API reference ---
    # "https://woocommerce.github.io/woocommerce-rest-api-docs/v3.html",
    ]
    return UnstructuredURLLoader(urls=URLS).load()
    

if __name__ == '__main__':
    print("oi")

    ingested_pdfs = ingest_pdfs()
    ingested_urls = ingest_urls()

    docs = ingested_pdfs + ingested_urls

    chunked_docs = [RecursiveCharacterTextSplitter(doc) for doc in docs]

    print(len(chunked_docs[0]))


