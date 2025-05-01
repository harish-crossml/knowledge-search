# Import required libraries
import fitz  # PyMuPDF for PDF reading
import re
import requests
import nltk
from nltk.corpus import stopwords
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer

# Initialize Qdrant vector database client
qdrant = QdrantClient(
    url="https://49b06d40-933c-4d23-a8cc-4afbf0cd8778.us-east-1-0.aws.cloud.qdrant.io:6333/",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.ALsLoEWB6dRmRrhDLazc9W8j2I_mZU_M8wLAZaaMhgI"
)

# Load sentence transformer model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# GROQ LLaMA API configuration
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Function to query LLaMA model using context and user question
def ask_llama(context, query):
    prompt = f"""
        You are an intelligent assistant helping users find answers from documents they uploaded. 
        Use only the information in the context below to answer the user's question. 
        If the answer is not in the context, say "I couldn't find that in the document." 
        Do not make up any information.

        Context:
        \"\"\"
        {context}
        \"\"\"

        Question: {query}

        Answer:"""
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "user", "content": prompt.strip()}
        ]
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=data)
    return response.json()['choices'][0]['message']['content']

# Function to create/reset a Qdrant collection for document storage
def setup_collection(collection_name="docs"):
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# Function to insert processed document chunks into Qdrant vector DB
def insert_documents(docs, collection_name="docs"):
    vectors = model.encode(docs).tolist()  # Create embeddings
    payload = [{"text": doc} for doc in docs]  # Associate each doc as payload
    qdrant.upsert(
        collection_name=collection_name,
        points=[
            {"id": i, "vector": vector, "payload": payload[i]}
            for i, vector in enumerate(vectors)
        ]
    )

# Function to search Qdrant for top-k similar document chunks based on query
def search(query, collection_name="docs", top_k=5):
    vector = model.encode(query).tolist()  # Convert query to embedding
    search_result = qdrant.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=top_k
    )
    return [hit.payload['text'] for hit in search_result]  # Return matching texts

# Download stopwords from NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Extract raw text from a PDF file
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Clean and preprocess text (remove punctuation, stopwords, lowercasing)
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)             # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)          # Remove punctuation
    text = text.lower()                          # Convert to lowercase
    return ' '.join([word for word in text.split() if word not in stop_words])

# Split long text into chunks for embedding and LLM processing
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
