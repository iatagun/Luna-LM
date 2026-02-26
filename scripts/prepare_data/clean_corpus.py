"""
Corpus Temizleme
Büyük .txt dosyalarını satır bazlı temizler.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import re


def clean_file(input_filename, output_filename):
    print(f"Cleaning {input_filename} -> {output_filename}...")
    
    with open(input_filename, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        
    print(f"Original line count: {len(lines)}")
    
    cleaned_lines = []
    split_pattern = re.compile(r'\s{4,}')
    seen_lines = set()

    for line in lines:
        line = line.strip()
        
        if not line:
            continue
            
        if line.startswith('#'):
            continue
            
        parts = split_pattern.split(line)
        
        for part in parts:
            part = part.strip()
            
            if not part:
                continue
            
            part = re.sub(r'\s+', ' ', part)
            
            if part in seen_lines:
                continue
            
            seen_lines.add(part)
            cleaned_lines.append(part)
            
    with open(output_filename, 'w', encoding='utf-8') as f:
        for line in cleaned_lines:
            f.write(line + '\n')
            
    print(f"Cleaned line count: {len(cleaned_lines)}")
    print("Done.")


if __name__ == "__main__":
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    clean_file(
        os.path.join(project_root, "foundation_corpus.txt"),
        os.path.join(project_root, "foundation_corpus_clean.txt")
    )
