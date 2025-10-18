import os, pathlib, textwrap, glob
import getpass
from langchain_community.document_loaders import UnstructuredURLLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
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
    return [[d] for d in UnstructuredURLLoader(urls=URLS).load()]
    

if __name__ == '__main__':
    print("oi")

    ingested_pdfs = ingest_pdfs()
    ingested_urls = ingest_urls()

    docs = ingested_pdfs + ingested_urls # List[List[Document]]

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    
    chunked_docs = [splitter.split_documents(doc) for doc in docs]
    all_docs = sum(chunked_docs, [])

    embedder = SentenceTransformerEmbeddings(model_name="thenlper/gte-small")

    db = FAISS.from_documents(all_docs, embedding=embedder)

    llm = Ollama(model="gemma3:1b")
    #response = llm.invoke("Explain what CrossFit is in one sentence.")
    #response = llm.invoke("How can AI be used for porn?")
    #print(response)

    #query = "How can AI be used for porn?"
    #results = db.similarity_search(query, k=4)    
    #embedding_vector = embedder.embed_query(chunked_docs)
    import pdb; pdb.set_trace()
    
    #print(len(embedding_vector))


