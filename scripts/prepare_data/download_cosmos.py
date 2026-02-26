"""
Cosmos Turkish Corpus'u indir, temizle ve corpus'a ekle
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from datasets import load_dataset
import re


def clean_text(text):
    """Temel metin temizleme"""
    if not text or not isinstance(text, str):
        return ""
    
    text = re.sub(r'\s+', ' ', text.strip())
    
    if len(text) < 20:
        return ""
    
    return text


def download_and_clean():
    print("Cosmos Turkish Corpus indiriliyor...")
    
    ds = load_dataset("ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0")
    
    print(f"Dataset yüklendi: {ds}")
    
    all_texts = []
    
    for split_name in ds.keys():
        print(f"\nProcessing split: {split_name}")
        split_data = ds[split_name]
        print(f"  Rows: {len(split_data)}")
        print(f"  Columns: {split_data.column_names}")
        
        text_col = None
        for col in ['text', 'content', 'sentence', 'document']:
            if col in split_data.column_names:
                text_col = col
                break
        
        if text_col is None:
            text_col = split_data.column_names[0]
            print(f"  Using first column: {text_col}")
        else:
            print(f"  Using column: {text_col}")
        
        for row in split_data:
            cleaned = clean_text(row[text_col])
            if cleaned:
                all_texts.append(cleaned)
    
    print(f"\nToplam temiz satır: {len(all_texts)}")
    
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    
    output_file = os.path.join(project_root, "cosmos_corpus_clean.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in all_texts:
            f.write(text + '\n')
    
    print(f"Kaydedildi: {output_file}")
    
    # Mevcut corpus ile birleştir
    print("\nMevcut corpus ile birleştiriliyor...")
    
    corpus_path = os.path.join(project_root, 'foundation_corpus_clean.txt')
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        existing = f.read()
    
    existing_lines = len(existing.strip().split('\n'))
    print(f"Mevcut corpus: {existing_lines} satır")
    
    with open(corpus_path, 'a', encoding='utf-8') as f:
        f.write('\n')
        for text in all_texts:
            f.write(text + '\n')
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        new_content = f.read()
    
    new_lines = len(new_content.strip().split('\n'))
    new_chars = len(new_content)
    
    print(f"\n=== YENİ CORPUS İSTATİSTİKLERİ ===")
    print(f"Toplam satır: {new_lines}")
    print(f"Toplam karakter: {new_chars:,}")
    print(f"Eklenen satır: {new_lines - existing_lines}")


if __name__ == "__main__":
    download_and_clean()
