import re
import os
import unicodedata
from pathlib import Path

input_folder = r'D:\Tisa(LLM)\dataset_Nepali'
output_file = r"D:\Tisa(LLM)\_nepali_corpus_v3_final.txt"

# input_folder = r'D:\Tisa(LLM)\test'
# output_file = r"D:\Tisa(LLM)\testing_output.txt"

DEVANAGARI_PATTERN = re.compile(r'[\u0900-\u097F]')
ENGLISH_PATTERN = re.compile(r'[a-zA-Z]')
URL_PATTERN = re.compile(r'http\S+|www\S+|https\S+')
EMAIL_PATTERN = re.compile(r'\S+@\S+')
PHONE_PATTERN = re.compile(r'\b\d{7,}\b')
YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')
ENGLISH_NUMBER_PATTERN = re.compile(r'\b[0-9]+\b')
# Updated: Don't match words inside < >
ENGLISH_WORD_PATTERN = re.compile(r'(?<![<])\b[a-zA-Z]+\b(?![>])')
UNWANTED_CHARS = re.compile(r'[^\u0900-\u097Fa-zA-Z0-9 <>।.,?!()\[\]\-\/\s]')
MULTIPLE_SPACES = re.compile(r'\s+')
SPACE_BEFORE_PUNCT = re.compile(r'\s+([।,.?!])')

def is_english_paragraph(text, threshold=0.7):
    """Check if paragraph is predominantly English"""
    eng = len(ENGLISH_PATTERN.findall(text))
    nep = len(DEVANAGARI_PATTERN.findall(text))
    
    if eng + nep == 0:
        return False
    
    return eng / (eng + nep) > threshold

def clean_nepali_text(text, min_length=10):
    """Clean and normalize Nepali text"""
    text = unicodedata.normalize('NFC', text)
    
    paragraphs = text.split('\n')
    cleaned_paragraphs = []
    
    for para in paragraphs:
        # Skip predominantly English paragraphs
        if is_english_paragraph(para):
            continue
        
        # Remove URLs and emails FIRST
        para = URL_PATTERN.sub(' ', para)
        para = EMAIL_PATTERN.sub(' ', para)
        
        # Remove unwanted characters BEFORE tokenization
        para = UNWANTED_CHARS.sub(' ', para)
        
        # NOW do tokenization on cleaned text
        para = PHONE_PATTERN.sub(' <PHONE> ', para)
        para = YEAR_PATTERN.sub(' <YEAR> ', para)
        para = ENGLISH_NUMBER_PATTERN.sub(' <NUM> ', para)  # Only English numbers
        # Nepali numbers are NOT replaced - they stay as-is
        para = ENGLISH_WORD_PATTERN.sub(' <ENG> ', para)
        
        # Normalize spacing
        para = SPACE_BEFORE_PUNCT.sub(r'\1', para)
        para = MULTIPLE_SPACES.sub(' ', para)
        para = para.strip()
        
        # Only keep paragraphs with minimum length
        if len(para) >= min_length:
            cleaned_paragraphs.append(para)
    
    return "\n".join(cleaned_paragraphs)

# Main processing
stats = {'files_processed': 0, 'files_skipped': 0, 'paragraphs_written': 0}

with open(output_file, "w", encoding="utf-8") as f_out:
    for file_path in Path(input_folder).rglob("*.txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f_in:
                text = f_in.read()
                cleaned_text = clean_nepali_text(text)
                
                if cleaned_text:
                    f_out.write(cleaned_text + "\n\n")
                    stats['paragraphs_written'] += cleaned_text.count('\n') + 1
                    stats['files_processed'] += 1
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            stats['files_skipped'] += 1

print(f"\n✓ Corpus saved to: {output_file}")
print(f"  Files processed: {stats['files_processed']}")
print(f"  Files skipped: {stats['files_skipped']}")
print(f"  Paragraphs written: {stats['paragraphs_written']}")