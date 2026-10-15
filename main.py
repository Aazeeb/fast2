from fastapi import FastAPI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load Groq API key from Colab Secrets
GROQ_API_KEY = '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'
GEMINI_API_KEy = 'AQ.Ab8RN6I705VMrF0-vQ282Y-1cyKlt4U6zYk_zZS2-_C4ZkT4SA'

# Load PDF
loader = PyPDFLoader("dell_latitude_5420_troubleshooting_demo.pdf")
pages = loader.load()

# Split PDF into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)
chunks = splitter.split_documents(pages)

# Create embeddings and FAISS database
embeddings = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001",api_key = GEMINI_API_KEY)

db = FAISS.from_documents(chunks, embeddings)

# Initialize Groq
model = ChatGroq(model="openai/gpt-oss-20b",api_key=GROQ_API_KEY)

# Prepare prompt
prompt = ChatPromptTemplate.from_template("""
Answer using only the context below.
If the answer is unavailable, say:
"I could not find that information in the manual."

Context:
{context}

Question:
{question}
""")

# RAG function
def ask_question(question):
    results = db.similarity_search(question, k=4)

    context = "\n\n".join(
        doc.page_content for doc in results
    )

    messages = prompt.format_messages(
        context=context,
        question=question
    )

    response = llm.invoke(messages)
    return response.content

# Create FastAPI application
app = FastAPI(title="Laptop Troubleshooting API")

@app.get("/")
def home():
    return {"message": "Laptop RAG API is running"}

@app.post("/ask")
def ask(question: str):
    return {
        "question": question,
        "answer": ask_question(question)
    }
