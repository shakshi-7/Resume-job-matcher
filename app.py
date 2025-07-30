import streamlit as st
from utils.file_loader import extract_text
from utils.embedder import get_embedding
from utils.matcher import cosine_similarity

st.set_page_config(page_title="Resume Matcher", layout="centered")
st.title("📄 AI-Powered Resume & Job Matcher")
st.write("Upload your resume and paste the job description to get a match score.")

resume_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
job_desc = st.text_area("Paste the Job Description", height=250)

if not resume_file:
    st.info("📂 Please upload a resume.")
if not job_desc:
    st.info("✍️ Paste the job description above.")

if resume_file and job_desc:
    with st.spinner("Analyzing..."):
        resume_text = extract_text(resume_file)
        if resume_text.strip() == "":
            st.error("No text found.")
        else:
            resume_vec = get_embedding(resume_text)
            job_vec = get_embedding(job_desc)
            score = cosine_similarity(resume_vec, job_vec) * 100

            st.subheader("🎯 Matching Score")
            st.metric("Fit %", f"{score:.2f}%")

            if score > 75:
                st.success("✅ Strong Match!")
            elif score > 50:
                st.warning("⚠️ Fair Match.")
            else:
                st.error("❌ Weak Match.")
