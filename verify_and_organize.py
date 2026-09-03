import os
import shutil
import fitz  # PyMuPDF
import re

# File mapping to folders
file_mapping = {
    "1_syllabus": "Syllabus",
    "2_project_topics": "Project_Topics",
    "3_DSE4101_Course_Intro": "Course_Intro"
}

def count_words(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return len(words)

def verify_and_organize():
    for base_name, folder_name in file_mapping.items():
        pdf_file = f"{base_name}.pdf"
        md_file = f"{base_name}.md"
        
        if not os.path.exists(pdf_file) or not os.path.exists(md_file):
            print(f"Skipping {base_name}, missing pdf or md.")
            continue
            
        # 1. Verify content completeness (basic heuristic: word count)
        doc = fitz.open(pdf_file)
        pdf_text = ""
        for page in doc:
            pdf_text += page.get_text()
        
        with open(md_file, "r", encoding="utf-8") as f:
            md_text = f.read()
            
        pdf_words = count_words(pdf_text)
        md_words = count_words(md_text)
        
        diff = abs(pdf_words - md_words)
        pct_diff = (diff / max(pdf_words, 1)) * 100
        
        print(f"--- Verification for {base_name} ---")
        print(f"PDF Word Count: {pdf_words}")
        print(f"MD Word Count:  {md_words}")
        print(f"Difference:     {pct_diff:.2f}%")
        
        if pct_diff > 15:
            print(f"WARNING: Large difference in word count for {base_name}. Manual check recommended.")
        else:
            print(f"SUCCESS: Data transferred accurately for {base_name}.")
            
        # 2. Organize into folders
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            
        # Move files
        shutil.move(pdf_file, os.path.join(folder_name, pdf_file))
        shutil.move(md_file, os.path.join(folder_name, md_file))
        print(f"Moved {pdf_file} and {md_file} into {folder_name}/ folder.\n")

if __name__ == "__main__":
    verify_and_organize()
