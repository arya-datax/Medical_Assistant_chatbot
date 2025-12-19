from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 



# 1. Extracting data from PDF file:
def load_pdf(data):
    loader = DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)

    document = loader.load()
    
    return document




# 2. Splitting data into Chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# 3. Download embedding model from hugging face
def download_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")