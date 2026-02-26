"""
Luna-LM Model Inference
Eğitilmiş modeli yükleyip metin üretimi yapar
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import glob

from luna.utils import load_model
from luna.generate import generate_text


# ==================== INFERENCE FONKSİYONLARI ====================

def interactive_mode(model, tokenizer, device, model_config):
    """İnteraktif mod — kullanıcıdan prompt al ve üret"""
    
    print("\n" + "="*60)
    print("LUNA-LM İNTERAKTİF MOD")
    print("="*60)
    print("\nKomutlar:")
    print("  - Metin girin ve Enter'a basın")
    print("  - 'quit' veya 'exit' yazarak çıkış yapın")
    print("  - 'params' yazarak parametreleri değiştirin")
    print("="*60 + "\n")
    
    # Varsayılan parametreler
    params = {
        'max_tokens': 100,
        'temperature': 0.8,
        'top_k': 50
    }
    
    while True:
        try:
            prompt = input("\n📝 Prompt: ").strip()
            
            if not prompt:
                continue
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\nGörüşmek üzere! 👋")
                break
            
            if prompt.lower() == 'params':
                print("\nMevcut parametreler:")
                print(f"  max_tokens: {params['max_tokens']}")
                print(f"  temperature: {params['temperature']}")
                print(f"  top_k: {params['top_k']}")
                
                try:
                    params['max_tokens'] = int(input("  Yeni max_tokens (Enter=değişmez): ") or params['max_tokens'])
                    params['temperature'] = float(input("  Yeni temperature (Enter=değişmez): ") or params['temperature'])
                    params['top_k'] = int(input("  Yeni top_k (Enter=değişmez): ") or params['top_k'])
                    print("  ✓ Parametreler güncellendi!")
                except:
                    print("  ✗ Geçersiz değer, parametreler değiştirilmedi.")
                continue
            
            # Metin üret
            print("\n🤖 Luna-LM:")
            generated = generate_text(
                model, tokenizer, device, prompt,
                max_new_tokens=params['max_tokens'],
                temperature=params['temperature'],
                top_k=params['top_k']
            )
            print(generated)
            
        except KeyboardInterrupt:
            print("\n\nGörüşmek üzere! 👋")
            break
        except Exception as e:
            print(f"\n❌ Hata: {e}")


# ==================== MAIN ====================

def main():
    print("\n" + "="*60)
    print("LUNA-LM INFERENCE")
    print("="*60 + "\n")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Proje kök dizini
    project_root = os.path.join(os.path.dirname(__file__), '..')
    
    # Checkpoint dizini bul (önce yeni yapı, sonra eski yapı)
    checkpoint_dirs = glob.glob(os.path.join(project_root, "checkpoints", "pretrain_*"))
    if not checkpoint_dirs:
        checkpoint_dirs = glob.glob(os.path.join(project_root, "luna_lm_checkpoints_*"))
    
    if not checkpoint_dirs:
        print("❌ Hiç checkpoint bulunamadı!")
        print("   Önce scripts/train.py ile model eğitin.")
        return
    
    # En son checkpoint'i seç
    checkpoint_dir = sorted(checkpoint_dirs)[-1]
    print(f"Checkpoint dizini: {checkpoint_dir}\n")
    
    # Modeli yükle
    model, tokenizer, model_config = load_model(checkpoint_dir, device=device)
    
    # Test prompts
    print("\n" + "="*60)
    print("TEST ÜRETİMLERİ")
    print("="*60)
    
    test_prompts = [
        "Bugün hava çok güzel",
        "Yapay zekâ teknolojisi",
        "Tarih boyunca insanlık",
        "Bilim ve teknoloji"
    ]
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        print("🤖 Luna-LM:")
        generated = generate_text(
            model, tokenizer, device, prompt,
            max_new_tokens=80,
            temperature=0.8,
            top_k=50
        )
        print(generated)
    
    # İnteraktif mod
    print("\n" + "="*60)
    use_interactive = input("\nİnteraktif moda geçmek ister misiniz? (y/n): ").strip().lower()
    
    if use_interactive == 'y':
        interactive_mode(model, tokenizer, device, model_config)
    else:
        print("\nBitti! İnteraktif mod için tekrar çalıştırın.")


if __name__ == "__main__":
    main()
