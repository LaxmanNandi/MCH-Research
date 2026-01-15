"""
Create PDF from markdown using fpdf2
"""

import os
import re
from fpdf import FPDF
import zipfile

BASE_DIR = "C:/Users/barla/mch_experiments/publication_analysis"
MD_FILE = os.path.join(BASE_DIR, "MCH_arXiv_Paper.md")
PDF_FILE = os.path.join(BASE_DIR, "MCH_arXiv_Paper.pdf")
ZIP_FILE = os.path.join(BASE_DIR, "MCH_Publication_Package.zip")

class PDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def chapter_title(self, title, level=1):
        if level == 0:
            self.set_font('Helvetica', 'B', 16)
            self.multi_cell(0, 10, title, align='C')
            self.ln(5)
        elif level == 1:
            self.set_font('Helvetica', 'B', 14)
            self.multi_cell(0, 8, title)
            self.ln(3)
        elif level == 2:
            self.set_font('Helvetica', 'B', 12)
            self.multi_cell(0, 7, title)
            self.ln(2)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        # Clean up text
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = text.encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 5, text)
        self.ln(2)

    def bullet_point(self, text):
        self.set_font('Helvetica', '', 10)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = text.encode('latin-1', 'replace').decode('latin-1')
        self.set_x(self.l_margin + 5)
        self.multi_cell(0, 5, "- " + text)

def clean_text(text):
    """Clean text for PDF compatibility."""
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = text.replace('ΔRCI', 'Delta-RCI')
    text = text.replace('Δ', 'Delta')
    text = text.replace('≥', '>=')
    text = text.replace('≤', '<=')
    text = text.replace('×', 'x')
    text = text.replace('—', '-')
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    text = text.encode('latin-1', 'replace').decode('latin-1')
    return text

def create_pdf():
    print("Creating PDF with fpdf2...")

    with open(MD_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    lines = content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line or line == '---':
            i += 1
            continue

        line = clean_text(line)

        # Main title
        if line.startswith('# ') and not line.startswith('## '):
            pdf.chapter_title(line[2:], level=0)

        # Section headers
        elif line.startswith('## '):
            pdf.ln(5)
            pdf.chapter_title(line[3:], level=1)

        # Subsection headers
        elif line.startswith('### '):
            pdf.ln(3)
            pdf.chapter_title(line[4:], level=2)

        # Bullet points
        elif line.startswith('- '):
            pdf.bullet_point(line[2:])

        # Numbered lists
        elif re.match(r'^\d+\. ', line):
            text = re.sub(r'^\d+\. ', '', line)
            pdf.set_font('Helvetica', '', 10)
            pdf.set_x(pdf.l_margin + 5)
            pdf.multi_cell(0, 5, text)

        # Skip complex tables, just show as text
        elif '|' in line:
            if '---' not in line:
                cells = [c.strip() for c in line.split('|') if c.strip()]
                if cells:
                    pdf.set_font('Helvetica', '', 9)
                    row_text = ' | '.join(cells)
                    pdf.multi_cell(0, 4, row_text)

        # Regular paragraphs
        else:
            if line:
                pdf.body_text(line)

        i += 1

    pdf.output(PDF_FILE)
    print(f"  Saved: {PDF_FILE}")

def update_zip():
    print("Updating ZIP with PDF...")

    files_to_zip = [
        "MCH_arXiv_Paper.md",
        "MCH_arXiv_Paper.docx",
        "MCH_arXiv_Paper.pdf",
        "mch_complete_analysis_report.txt",
        "figure1_response_coherence.png",
        "figure2_effect_sizes.png",
        "figure3_vendor_tier.png",
    ]

    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename in files_to_zip:
            filepath = os.path.join(BASE_DIR, filename)
            if os.path.exists(filepath):
                zipf.write(filepath, filename)
                print(f"  Added: {filename}")

    print(f"  Saved: {ZIP_FILE}")

if __name__ == "__main__":
    create_pdf()
    update_zip()
    print("\nDone! All files created.")
