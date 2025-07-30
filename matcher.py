
import pdfplumber
import docx2txt
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_resume_text(file_path):
    if file_path.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        return text
    elif file_path.endswith(".docx"):
        return docx2txt.process(file_path)
    else:
        return ""

def get_jobs_from_google(job_title, location="India", num_results=10):
    api_key = "0aa45283f30a3d1d4f355652070b41fa871858ded87aa5a8594c1ebdcfbb1a76"
    params = {
        "engine": "google_jobs",
        "q": job_title,
        "location": location,
        "api_key": api_key,
    }
    response = requests.get("https://serpapi.com/search", params=params)
    data = response.json()
    jobs = []
    if "jobs_results" in data:
        for job in data["jobs_results"][:num_results]:
            jobs.append({
                "title": job["title"],
                "company": job.get("company_name", ""),
                "description": job.get("description", ""),
                "link": job.get("apply_link", "#")
            })
    return jobs

def rank_jobs(resume_text, jobs):
    corpus = [resume_text] + [job["description"] for job in jobs]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    resume_vec = tfidf_matrix[0]
    job_vecs = tfidf_matrix[1:]
    scores = cosine_similarity(resume_vec, job_vecs).flatten()
    ranked_jobs = sorted(zip(jobs, scores), key=lambda x: x[1], reverse=True)
    return ranked_jobs
