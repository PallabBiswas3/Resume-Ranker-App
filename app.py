import streamlit as st
import fitz  # PyMuPDF
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ---------- OCR Environment Setup ----------
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ---------- TEXT EXTRACTION FUNCTION ----------
def extract_text_from_pdf(file):
    file.seek(0)
    file_bytes = file.read()

    # Step 1: Try normal text extraction using PyMuPDF
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    # Step 2: Fallback to OCR if text is too short and OCR is available
    if len(text.strip()) < 100:
        if OCR_AVAILABLE:
            st.warning(f"⚠️ Low text in {file.name}. Using OCR fallback.")
            images = convert_from_bytes(file_bytes)
            ocr_text = ""
            for image in images:
                ocr_text += pytesseract.image_to_string(image)
            return ocr_text
        else:
            st.warning(f"⚠️ Low text in {file.name}, and OCR is not available on this server.")
            return ""

    return text

# ---------- RESUME SECTION EXTRACTION ----------
def extract_resume_sections(text):
    lines = text.split('\n')
    summary, experience, education, certifications, skills = [], [], [], [], []
    current_section = None

    for line in lines:
        line_strip = line.strip()
        if re.match(r'\bSummary\b', line_strip, re.I):
            current_section = 'summary'; continue
        elif re.match(r'\b(Highlights|Accomplishments)\b', line_strip, re.I):
            current_section = 'summary'; continue
        elif re.match(r'\bExperience\b', line_strip, re.I):
            current_section = 'experience'; continue
        elif re.match(r'\bEducation\b', line_strip, re.I):
            current_section = 'education'; continue
        elif re.match(r'\bCertifications\b', line_strip, re.I):
            current_section = 'certifications'; continue
        elif re.match(r'\bSkills\b', line_strip, re.I):
            current_section = 'skills'; continue
        elif re.match(r'\bInterests\b|\bAdditional Information\b', line_strip, re.I):
            current_section = None; continue

        if current_section == 'summary':
            summary.append(line_strip)
        elif current_section == 'experience':
            if re.search(r'\b\d{4}\b|\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', line_strip, re.I):
                experience.append(line_strip)
        elif current_section == 'education':
            education.append(line_strip)
        elif current_section == 'certifications':
            certifications.append(line_strip)
        elif current_section == 'skills':
            skills.append(line_strip)

    return {
        "summary": ' '.join(summary),
        "experience": ' '.join(experience),
        "education": ' '.join(education),
        "certifications": ' '.join(certifications),
        "skills": ' '.join(skills)
    }

# ---------- RANKING FUNCTION ----------
def rank_resumes(jd_text, resume_data):
    jd_blob = jd_text.lower()
    corpus = [jd_blob]
    names = []

    for name, sections in resume_data:
        text_blob = (sections['summary'] + ' ' + sections['experience'] + ' ' + sections['skills']).lower()
        corpus.append(text_blob)
        names.append(name)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    ranked = sorted(zip(names, similarities), key=lambda x: x[1], reverse=True)
    return ranked

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Resume Ranker", page_icon="📄")
st.title("📄 AI Resume Ranker (Smart OCR Fallback)")

st.info("Upload a Job Description PDF and multiple Resume PDFs to rank them based on content relevance. Scanned PDFs will be OCR-processed only if supported by this server.")

jd_file = st.file_uploader("📥 Upload Job Description (PDF)", type=["pdf"])
resume_files = st.file_uploader("📥 Upload Multiple Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

if st.button("🚀 Rank Resumes"):
    if not jd_file or not resume_files:
        st.warning("Please upload both a job description and at least one resume.")
    else:
        jd_text = extract_text_from_pdf(jd_file)
        if not jd_text.strip():
            st.error("❌ Job Description PDF could not be read. Try another file.")
        else:
            resume_data = []
            for file in resume_files:
                text = extract_text_from_pdf(file)
                if not text.strip():
                    st.warning(f"❌ Skipping {file.name} — no readable content found.")
                    continue
                sections = extract_resume_sections(text)
                resume_data.append((file.name, sections))

            if not resume_data:
                st.error("❌ No valid resumes to rank.")
            else:
                ranked_resumes = rank_resumes(jd_text, resume_data)

                st.subheader("🏆 Ranked Resumes")
                for i, (name, score) in enumerate(ranked_resumes, 1):
                    st.write(f"**{i}. {name}** — Score: {score:.4f}")

                # CSV Download
                df_out = pd.DataFrame(ranked_resumes, columns=["Resume Name", "Match Score"])
                csv = df_out.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Results as CSV", csv, "ranked_resumes.csv", "text/csv")
