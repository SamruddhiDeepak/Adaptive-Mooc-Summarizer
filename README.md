# Adaptive MOOC Summarizer, Flashcard Generator & MCQ Quiz Builder

This project is a **Flask-based backend service** that takes a PDF file as input and provides:

**Clean PDF summaries (module-wise)**
**Auto-generated flashcards**
**Auto-generated MCQ quiz**
**Chunk-based summarization using TextRank**

---

## Features

### 🔹 **1. PDF Text Extraction**

* Uses `PyPDF2` to extract readable text from uploaded PDF files.
* Cleans and preprocesses extracted text.

### 🔹 **2. Automatic Text Chunking**

* Large PDF text is split into chunks of **2500 characters**.
* Each chunk becomes a module in the final summary.

### 🔹 **3. TextRank Summarization**

* Summarizes each chunk using **TF-IDF + cosine similarity + PageRank**.
* Produces high-quality multi-sentence summaries.

### 🔹 **4. Module Title Generation**

* Uses OpenRouter LLM to generate short, highly relevant module titles.


### 🔹 **5. Flashcard Generation**

* Uses OpenRouter LLM to create structured JSON flashcards.

### 🔹 **6. MCQ Quiz Generator**

* Creates multiple-choice questions with:

  * Exactly 4 options
  * One correct answer
  * JSON output format

### 🔹 **7. CORS Enabled**

* Works easily with any frontend (React, Vue, Angular, plain JS).

---

## 🛠️ Tech Stack

| Component         | Technology                   |
| ----------------- | ---------------------------- |
| Backend Framework | Flask                        |
| Text Extraction   | PyPDF2                       |
| Summarization     | TextRank (TF-IDF + PageRank) |
| PDF Output        | ReportLab                    |
| LLM Integration   | OpenRouter API               |
| NLP               | scikit-learn                 |

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <your-repo-url>
cd <project-folder>
```

Create & activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=google/gemini-2.0-flash-001
```

---

## ▶️ Run the Server

```bash
python app.py
```

The server starts at:

```
http://127.0.0.1:5000
```

---

## 🔮 Future Enhancements 

* Add support for image-based PDFs via OCR
* Support more quiz types (true/false, fill-in-the-blank)

---

## 📝 License

MIT License. Free to use and modify.


