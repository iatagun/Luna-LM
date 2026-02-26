"""
Alpaca Turkish Combined Dataset'i indir ve JSONL formatına dönüştür.
Kaynak: https://huggingface.co/datasets/cenfis/alpaca-turkish-combined
82,353 soru-cevap çifti (Türkçe)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
from datasets import load_dataset


OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "sft_dataset.jsonl")


def download_and_convert():
    print("="*60)
    print("ALPACA TURKISH COMBINED DATASET")
    print("="*60)
    print("\nKaynak: cenfis/alpaca-turkish-combined")
    print("İndiriliyor...\n")
    
    # HuggingFace'den indir
    ds = load_dataset("cenfis/alpaca-turkish-combined")
    
    train_data = ds["train"]
    print(f"✓ Dataset yüklendi: {len(train_data):,} satır")
    print(f"  Sütunlar: {train_data.column_names}")
    
    # JSONL formatına dönüştür
    print(f"\nJSONL'e dönüştürülüyor: {OUTPUT_FILE}")
    
    valid = 0
    skipped = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for i, row in enumerate(train_data):
            instruction = (row.get("instruction") or "").strip()
            input_text = (row.get("input") or "").strip()
            output_text = (row.get("output") or "").strip()
            
            # instruction ve output zorunlu
            if not instruction or not output_text:
                skipped += 1
                continue
            
            # user = instruction + input (varsa)
            user_text = instruction
            if input_text:
                user_text += f"\n{input_text}"
            
            entry = {
                "user": user_text,
                "assistant": output_text
            }
            
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            valid += 1
            
            if (i + 1) % 20000 == 0:
                print(f"  İşlenen: {i+1:,}/{len(train_data):,}...")
    
    print(f"\n{'='*60}")
    print(f"TAMAMLANDI!")
    print(f"{'='*60}")
    print(f"  ✓ Geçerli örnek: {valid:,}")
    print(f"  ✗ Atlanan:       {skipped:,}")
    print(f"  📄 Dosya: {OUTPUT_FILE}")
    
    # Dosya boyutu
    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"  💾 Boyut: {size_mb:.1f} MB")
    
    # Örnek göster
    print(f"\n{'='*60}")
    print("İLK 3 ÖRNEK:")
    print(f"{'='*60}")
    
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            entry = json.loads(line)
            print(f"\n[{i+1}] User: {entry['user'][:80]}...")
            print(f"    Asst: {entry['assistant'][:80]}...")
    
    print(f"\n\nSFT eğitimini başlatmak için:")
    print(f"  python sft/train_sft.py")


if __name__ == "__main__":
    download_and_convert()
