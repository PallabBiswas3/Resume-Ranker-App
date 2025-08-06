# 📄 AI Resume Ranker

A Streamlit web app that ranks candidate resumes against a job description using TF-IDF and cosine similarity. It automates the process of shortlisting candidates by analyzing the relevance of their resumes to the given job requirements.

---

## 🔍 Features

* **PDF Text Extraction**: Extracts text from PDFs using PyMuPDF (no OCR).
* **Section Parsing**: Identifies and parses key resume sections: Summary, Experience, Education, Certifications, and Skills.
* **Relevance Scoring**: Computes TF-IDF vectors and cosine similarity scores between a job description and each resume.
* **Interactive UI**: Upload job descriptions and multiple resumes via a Streamlit interface.
* **Result Export**: Download ranking results as a CSV file for easy sharing and record-keeping.

---

## 🚀 Quick Start

### Prerequisites

* Python 3.7 or higher
* [pip](https://pip.pypa.io/en/stable/) package manager

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/ai-resume-ranker.git
   cd ai-resume-ranker
   ```

2. **Create and activate a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate     # macOS/Linux
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   **`requirements.txt`** should include:

   ```text
   streamlit
   PyMuPDF
   pandas
   scikit-learn
   ```

### Running the App

```bash
streamlit run app.py
```

1. Open the displayed URL in your browser (usually `http://localhost:8501`).
2. Upload a **Job Description** PDF and multiple **Resume** PDFs.
3. Click **`🚀 Rank Resumes`** to see ranked results and download a CSV report.

---

## 📂 Project Structure

```
ai-resume-ranker/
├── app.py             # Main Streamlit application
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
└── LICENSE            # (Optional) license file
```

---

## 🛠 How It Works

1. **Text Extraction**

   * Uses PyMuPDF (`fitz`) to load each PDF and extract raw text.
   * Skips image-based PDFs if extracted text length is too low.

2. **Section Extraction**

   * Splits the extracted text into lines and categorizes them under headings like `Summary`, `Experience`, etc.
   * Applies regex matching to identify date lines in the Experience section.

3. **Ranking Algorithm**

   * Concatenates the JD text and each resume's key sections into a corpus.
   * Vectorizes the corpus using TF-IDF.
   * Computes cosine similarity between the JD vector and resume vectors.
   * Sorts and displays resumes by descending similarity score.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check \[issues page] and submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

