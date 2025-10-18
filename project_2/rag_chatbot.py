import os, pathlib, textwrap, glob
import getpass
from langchain_community.document_loaders import UnstructuredURLLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.llms import Ollama
from langchain.chains import create_history_aware_retriever
from langchain.prompts import PromptTemplate

print("✅ Libraries imported! You're good to go!")

SYSTEM_TEMPLATE = """
    You are a **Customer Support Chatbot**. Use only the information in CONTEXT to answer.
    If the answer is not in CONTEXT, respond with “I'm not sure from the docs.”

    Rules:
    1) Use ONLY the provided <context> to answer.
    2) If the answer is not in the context, say: "I don't know based on the retrieved documents."
    3) Be concise and accurate. Prefer quoting key phrases from the context.
    4) When possible, cite sources as using the metadata.

    INPUT:
    {input}
    """

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
    ingested_pdfs = ingest_pdfs()
    ingested_urls = ingest_urls()

    docs = ingested_pdfs + ingested_urls # List[List[Document]]

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    
    chunked_docs = [splitter.split_documents(doc) for doc in docs]
    all_docs = sum(chunked_docs, [])

    embedder = SentenceTransformerEmbeddings(model_name="thenlper/gte-small")

    faiss_index = FAISS.from_documents(all_docs, embedding=embedder)
    retriever = faiss_index.as_retriever(search_kwargs={"k": 4})

    llm = Ollama(model="gemma3:1b", temperature=0.1)
    #response = llm.invoke("Explain what CrossFit is in one sentence.")
    #response = llm.invoke("How can AI be used for porn?")
    #print(response)

    #query = "How can AI be used for porn?"
    #results = db.similarity_search(query, k=4)    
    #embedding_vector = embedder.embed_query(chunked_docs)

    prompt_template = contextualize_prompt = PromptTemplate(
        template=
            SYSTEM_TEMPLATE
        )
    chain = create_history_aware_retriever(llm=llm, retriever=retriever, prompt=prompt_template)

    test_questions = [
        "If I'm not happy with my purchase, what is your refund policy and how do I start a return?",
        "How long will delivery take for a standard order, and where can I track my package once it ships?",
        "What's the quickest way to contact your support team, and what are your operating hours?",
    ]

    past_questions = ""
    for question in test_questions:
        response = chain.invoke({"input": question, "chat_history": past_questions})
        print("-------------------------------------------------------")
        print(response)
        past_questions += f"\nQuestion: {question}, Response: {response}"
    


