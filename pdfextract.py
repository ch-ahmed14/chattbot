from PyPDF2 import PdfReader
import pdfplumber
import fitz
import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def extract_text_from_pdf(pdf_path):
    all_text = ""

    # Extract text using PyPDF2
    reader = PdfReader(pdf_path)
    print(len(reader.pages))
    for i in range(len(reader.pages)):
        page = reader.pages[i]
        all_text += page.extract_text()

    # Extract tables using pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        for i in pdf.pages:
            tables = i.extract_tables()
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

if __name__ == "__main__":
    pdf_path = r'C:\Users\pc\Desktop\work of stage\file - Copy.pdf'
    cleaned_text = extract_text_from_pdf(pdf_path)
    with open('cleaned_text.txt', 'w', encoding='utf-8') as f:
        f.write(cleaned_text)