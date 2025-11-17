import os
import re
import json
import random
import requests
from io import BytesIO
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Config / Globals
app = Flask(__name__)
CORS(app)

MAX_CHUNK_CHARS = 2500
MAX_SUMMARY_SENTENCES = 6
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")

# Text preprocessing & summarization
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return sentences

def textrank_summarize(sentences, max_sentences=MAX_SUMMARY_SENTENCES):
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    sim_matrix = cosine_similarity(X, X)
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    top_sentences = [s for score, s in ranked_sentences[:max_sentences]]
    top_sentences.sort(key=lambda s: sentences.index(s))
    return " ".join(top_sentences)

def summarize_text(text, max_sentences=MAX_SUMMARY_SENTENCES):
    sentences = preprocess_text(text)
    return textrank_summarize(sentences, max_sentences=max_sentences)

#PDF text extraction
def extract_text_from_pdf(file):
    try:
        pdf_reader = PdfReader(file)
        text_parts = [page.extract_text() or "" for page in pdf_reader.pages]
        return "\n\n".join([p for p in text_parts if p.strip()]).strip()
    except Exception as e:
        print("[ERROR] Failed to extract PDF text:", e)
        return ""

#Create clean PDF
def create_clean_pdf(sections):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    title_style = styles['Title']
    title_style.alignment = TA_CENTER
    story.append(Paragraph("📘 PDF Summary", title_style))
    story.append(Spacer(1, 1*cm))

    heading_style = styles['Heading2']
    body_style = styles['Normal']

    for section in sections:
        story.append(Paragraph(section['title'], heading_style))
        story.append(Spacer(1, 0.3*cm))
        for line in section['content']:
            story.append(Paragraph(line, body_style))
            story.append(Spacer(1, 0.2*cm))
        story.append(PageBreak())

    doc.build(story)
    buffer.seek(0)
    return buffer

#OpenRouter robust parsing
def parse_openrouter_response(resp_text):
    try:
        clean_text = re.sub(r"^```json|```$", "", resp_text.strip(), flags=re.MULTILINE).strip()
        return json.loads(clean_text)
    except json.JSONDecodeError:
        flashcards = []
        matches = re.findall(r'\{.*?\}', resp_text, re.DOTALL)
        for m in matches:
            try:
                flashcards.append(json.loads(m))
            except:
                continue
        return flashcards

#Generate Flashcards
def generate_flashcards_from_text(text, num_flashcards=15):
    if not OPENROUTER_API_KEY:
        print("[WARN] OPENROUTER_API_KEY not set")
        return []

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

    prompt = (
        f"Generate {num_flashcards} flashcards from the following text. "
        "Return ONLY a JSON list in this format: "
        "[{\"question\": \"...\", \"answer\": \"...\"}].\n\n"
        f"Text:\n{text[:4000]}"
    )

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 2000
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        raw = data["choices"][0]["message"]["content"].strip()
        flashcards = parse_openrouter_response(raw)
        return flashcards
    except Exception as e:
        print("[WARN] Flashcards generation failed:", e)
        return []

#Generate MCQ Quiz
def generate_mcq_quiz_from_text(text, num_questions=10):
    if not OPENROUTER_API_KEY:
        print("[WARN] OPENROUTER_API_KEY not set")
        return []

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

    prompt = (
        f"Generate {num_questions} multiple-choice questions from the text. "
        "Each question must have exactly 4 options and one correct answer. "
        "Return ONLY a JSON array in this format:\n"
        "[{\"question\": \"...\", \"options\": [\"...\", \"...\", \"...\", \"...\"], \"answer\": \"...\"}]\n\n"
        f"Text:\n{text[:4000]}"
    )

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 2500
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        raw = data["choices"][0]["message"]["content"].strip()
        quiz = parse_openrouter_response(raw)

        for q in quiz:
            options = q['options']
            random.shuffle(options)
            q['options'] = options

        return quiz
    except Exception as e:
        print("[WARN] MCQ quiz generation failed:", e)
        return []

def generate_module_title(chunk_text, max_words=6):
    if not OPENROUTER_API_KEY:
        print("[WARN] OPENROUTER_API_KEY not set")
        return "Overview"

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "user",
             "content": (
                 "Generate a short descriptive title (3-6 words) for the following content. "
                 "Return only the title, no extra text or punctuation:\n\n" + chunk_text
             )}
        ],
        "temperature": 0.5,
        "max_tokens": 16
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        title = data["choices"][0]["message"]["content"].strip()
        title_words = title.split()[:max_words]
        return " ".join(title_words) or "Overview"
    except Exception as e:
        print("[WARN] OpenRouter title generation failed:", e)
        return "Overview"

# Flask Endpoint
@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    text = extract_text_from_pdf(file)
    if not text.strip():
        return jsonify({"error": "No readable text found in PDF"}), 400

    chunks = []
    current_chunk = ""
    for paragraph in text.split("\n\n"):
        if len(current_chunk) + len(paragraph) + 1 > MAX_CHUNK_CHARS:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n"
        else:
            current_chunk += paragraph + "\n"
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    module_summaries = []
    for idx, chunk in enumerate(chunks, 1):
        summary = summarize_text(chunk)
        module_title = generate_module_title(chunk)
        module_summaries.append({"title": f"Module {idx}: {module_title}", "content": [summary]})


    pdf_buffer = create_clean_pdf(module_summaries)

    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=False,
        download_name="summary.pdf"
    )

@app.route("/flashcards", methods=["POST"])
def flashcards_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    text = extract_text_from_pdf(file)
    flashcards = generate_flashcards_from_text(text, num_flashcards=15)
    return jsonify({"flashcards": flashcards})

@app.route("/quiz", methods=["POST"])
def quiz_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    text = extract_text_from_pdf(file)
    quiz = generate_mcq_quiz_from_text(text, num_questions=10)
    return jsonify({"quiz": quiz})

@app.route("/")
def index():
    return "PDF Summarizer & OpenRouter Flashcards/Quiz API is running."

if __name__ == "__main__":
    app.run(debug=True)
