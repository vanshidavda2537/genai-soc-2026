from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

#loading pdf

pdf_path="sample.pdf"

loader=PyPDFLoader(pdf_path)
documents=loader.load()
print(f"Loaded {len(documents)} pages from PDF")

#chunking 

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks=text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

#embedding

embedding_model=HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

#store in chromdb

vector_store=Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_store"
)

print(
     f"\nIndexed {len(chunks)} chunks into ChromaDB "
    f"(persistent directory: ./chroma_store)"
)

#printing first 3 chunks

print("\n SAMPLE CHUNKS: ")

for i in range(min(3,len(chunks))):
    chunk=chunks[i]
    page_number=chunk.metadata.get("page","Unknown")

    print(f"\n Sample Chunk {i+1}")
    print(f"Page Number: {page_number + 1}")

    print("/n Content Preview: ")
    print("-"*50)

    print(chunk.page_content[:300])
    print("/n"+"="*60)

