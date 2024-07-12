import re
import pdfplumber
from PyPDF2 import PdfReader
import fitz

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def extract_text_from_pdf(pdf_path):
    all_text = ""

    # Extract text using PdfReader from PyPDF2
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        all_text += page.extract_text()

    # Extract tables using pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                all_text += str(table)

    # Extract text using fitz (PyMuPDF)
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        all_text += text

    cleaned_text = preprocess_text(all_text)
    return cleaned_text
