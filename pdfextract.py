from PyPDF2 import PdfReader
import pdfplumber
import fitz
reader = PdfReader('C:\\Users\\pc\\Desktop\\work of stage\\file - Copy.pdf')
print(len(reader.pages))
for i in range(len(reader.pages)):
    page = reader.pages[i]
    print(page.extract_text())
for i in page.images:
    with open(i.name, 'wb') as f:
        f.write(i.data)
with pdfplumber.open('C:\\Users\\pc\\Desktop\\work of stage\\file - Copy.pdf') as pdf:
    for i in pdf.pages:
        tables = i.extract_tables()
        for table in tables:
            print(table)
doc = fitz.open('C:\\Users\\pc\\Desktop\\work of stage\\file - Copy.pdf')

all = ""
for page_num in range(len(doc)):
    page = doc.load_page(page_num) 
    text = page.get_text()  
    all+= f"Page {page_num + 1}:\n{text}\n"
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(all)
