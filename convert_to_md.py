import pymupdf4llm
import glob
import os

pdf_files = glob.glob("*.pdf")
for pdf_file in pdf_files:
    print(f"Converting {pdf_file}...")
    try:
        md_text = pymupdf4llm.to_markdown(pdf_file)
        base_name = os.path.splitext(pdf_file)[0]
        md_file = f"{base_name}.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"Successfully converted {pdf_file} to {md_file}")
    except Exception as e:
        print(f"Error converting {pdf_file}: {e}")
