import streamlit as st
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import PyPDF2
import uvicorn
import io

nltk.download('punkt')
nltk.download('stopwords')

app = FastAPI()

# Enable CORS for all domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Preprocess text (same as before)
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(tokens)

# Match job description with resume (same as before)
def match_job_description(job_description, resume_text):
    preprocessed_job_description = preprocess_text(job_description)
    preprocessed_resume_text = preprocess_text(resume_text)

    vectorizer = TfidfVectorizer()
    job_description_vector = vectorizer.fit_transform([preprocessed_job_description])
    resume_vector = vectorizer.transform([preprocessed_resume_text])

    similarity_score = cosine_similarity(job_description_vector, resume_vector)[0][0]
    return similarity_score

# Endpoint for uploading job description and resume
@app.post("/upload")
async def upload_file(job_description: str = Form(...), resume: UploadFile = Form(...)):
    resume_text = ""

    if resume.filename.endswith('.txt'):
        resume_text = await resume.read()
        resume_text = resume_text.decode('utf-8')
    elif resume.filename.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(await resume.read()))
        for page_num in range(len(pdf_reader.pages)):
            resume_text += pdf_reader.pages[page_num].extract_text()

    similarity_score = match_job_description(job_description, resume_text)
    return {"similarity_score": similarity_score}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8600)
