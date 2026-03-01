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


def download_and_convert(limit=1000000):
    print("="*60)
    print("INSTRUCTURCA TURKISH DATASET")
    print("="*60)
    print("\nKaynak: turkish-nlp-suite/InstrucTurca")
    print(f"Hedef: İlk {limit:,} satır\n")
    
    # HuggingFace'den indir
    ds = load_dataset("turkish-nlp-suite/InstrucTurca")
    
    train_data = ds["train"]
    print(f"✓ Dataset yüklendi: {len(train_data):,} satır")
    
    # JSONL formatına dönüştür
    print(f"\nJSONL'e dönüştürülüyor: {OUTPUT_FILE}")
    
    valid = 0
    skipped = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for i, row in enumerate(train_data):
            if valid >= limit:
                break
            
            # Case-insensitive bir sözlük oluştur (anahtarları küçük harfe çevir)
            row_lower = {k.lower(): v for k, v in row.items()}

            if i == 0:
                print(f"  🔍 Örnek Kolonlar: {list(row.keys())}")

            # Esnek Kolon Eşleştirme (Küçük harf üzerinden)
            instruction = ""
            input_text = ""
            output_text = ""

            # Instruction / Prompt adayları
            for key in ["instruction", "prompt", "question", "user", "input_text", "query", "input"]:
                if row_lower.get(key):
                    instruction = str(row_lower[key]).strip()
                    break
            
            # Input / Context adayları (isteğe bağlı - ek bilgi varsa)
            # Eğer 'input' hem ana soru hem ek bilgi olarak kullanılıyorsa çakışabilir
            # InstrucTurca'da 'Input' ana soru gibi duruyor.
            for key in ["context", "context_text"]:
                if row_lower.get(key) and str(row_lower[key]).lower() != "none":
                    input_text = str(row_lower[key]).strip()
                    break

            # Output / Response adayları
            for key in ["output", "response", "assistant", "answer", "completion", "target"]:
                if row_lower.get(key):
                    output_text = str(row_lower[key]).strip()
                    break

            if not instruction or not output_text:
                skipped += 1
                if skipped < 5:
                    print(f"  ⚠️ Atlanan satır {i} - Mevcut: {list(row.keys())}")
                continue
            
            user_text = instruction
            if input_text:
                user_text += f"\n{input_text}"
            
            entry = {
                "user": user_text,
                "assistant": output_text
            }
            
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            valid += 1
            
            if (i + 1) % 50000 == 0:
                print(f"  İşlenen: {i+1:,}...")
    
    print(f"\n{'='*60}")
    print(f"TAMAMLANDI!")
    print(f"{'='*60}")
    print(f"  ✓ Geçerli örnek: {valid:,}")
    print(f"  ✗ Atlanan:       {skipped:,}")
    print(f"  📄 Dosya: {OUTPUT_FILE}")
    
    # Dosya boyutu
    if os.path.exists(OUTPUT_FILE):
        size_gb = os.path.getsize(OUTPUT_FILE) / (1024**3)
        print(f"  💾 Boyut: {size_gb:.2f} GB")
    
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
    download_and_convert(limit=1000000)
