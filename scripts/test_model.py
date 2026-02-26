"""
Luna-LM Test Script
Eğitilmiş modeli (pretrained veya SFT) test etmek için
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import glob

from luna.utils import load_model
from luna.generate import generate_text


# SFT Prompt Format
SYSTEM_PROMPT = "Senin adın Luna. Amacın insanlara yardımcı olmak ve sorulara açık, anlaşılır cevaplar vermektir. Emin olmadığın konularda bunu belirtir, uydurma bilgi eklemezsin. Cevaplarını nazik, sade ve doğal bir Türkçe ile yazarsın."


def format_sft_prompt(user_query):
    return f"<system>{SYSTEM_PROMPT}</system>\n<user>{user_query}</user>\n<assistant>"


def clean_sft_output(generated):
    """Tokenizer artifactlerini temizle ve asistan cevabını çıkar"""
    clean_gen = generated.replace("< assistant >", "<assistant>").replace("< / assistant >", "</assistant>")
    clean_gen = clean_gen.replace("< user >", "<user>").replace("< / user >", "</user>")
    clean_gen = clean_gen.replace("< system >", "<system>").replace("< / system >", "</system>")
    
    # En son açılan <assistant> taginden sonrasını al
    if "<assistant>" in clean_gen:
        answer = clean_gen.split("<assistant>")[-1]
    else:
        answer = clean_gen
        
    # Stop token kontrolü
    for stop_token in ["</assistant>", "<user>", "<system>", "[SEP]"]:
        if stop_token in answer:
            answer = answer.split(stop_token)[0]
    
    # Gereksiz karakter temizliği
    answer = answer.strip()
    while answer and (answer[0] in ('>', ' ', '.')):
        answer = answer[1:].strip()
        
    return answer


def find_best_checkpoint(project_root):
    """En iyi modeli bul: önce SFT, yoksa pretrained"""
    
    # 1. SFT checkpoint (en son)
    sft_dirs = sorted(glob.glob(os.path.join(project_root, "checkpoints", "sft_*")))
    if sft_dirs:
        latest_sft = sft_dirs[-1]
        best_model = os.path.join(latest_sft, "best_sft_model.pt")
        if os.path.exists(best_model):
            print(f"  ✓ SFT model bulundu: {latest_sft}")
            return latest_sft, "sft"
    
    # 2. Pretrained checkpoint (yeni yapı)
    pretrain_dirs = sorted(glob.glob(os.path.join(project_root, "checkpoints", "pretrain_*")))
    if pretrain_dirs:
        print(f"  ⚠ SFT model yok, pretrained kullanılacak")
        return pretrain_dirs[-1], "pretrained"
    
    # 3. Eski yapı
    old_dirs = sorted(glob.glob(os.path.join(project_root, "luna_lm_checkpoints_*")))
    if old_dirs:
        print(f"  ⚠ SFT model yok, eski pretrained kullanılacak")
        return old_dirs[-1], "pretrained"
    
    return None, None


def main():
    print("="*60)
    print("LUNA-LM TEST")
    print("="*60)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Proje kök dizini
    project_root = os.path.join(os.path.dirname(__file__), '..')
    
    # Model bul ve yükle
    print("\n📦 Model aranıyor...")
    checkpoint_path, model_type = find_best_checkpoint(project_root)
    
    if checkpoint_path is None:
        print("❌ Hiç model bulunamadı!")
        print("   Önce eğitim yapın: python scripts/train.py")
        return
    
    print(f"  Model tipi: {model_type.upper()}")
    print(f"  Yol: {checkpoint_path}")
    
    try:
        model, tokenizer, config = load_model(checkpoint_path, device)
    except Exception as e:
        print(f"\n⚠️ Model yüklenirken hata: {e}")
        return
    
    # Test soruları
    print("\n" + "="*60)
    print("SFT METİN ÜRETİMİ TESTİ")
    print("="*60)
    
    test_questions = [
        "Güneş hangi yönden doğar?",
        "Ampulü kim buldu?",
        "İstanbul'un önemi nedir?",
        "Mutluluk nedir?",
        "Yapay zeka ne işe yarar?",
        "Türkiye'nin başkenti neresidir?",
    ]
    
    for q in test_questions:
        print(f"\n❓ {q}")
        
        full_prompt = format_sft_prompt(q)
        
        generated = generate_text(
            model, tokenizer, device,
            full_prompt, 
            max_new_tokens=150, 
            temperature=0.3,    
            top_k=40,
            repetition_penalty=1.2
        )
        
        answer = clean_sft_output(generated)
        print(f"🤖 {answer}")

    # İnteraktif mod
    print("\n" + "="*60)
    print("İNTERAKTİF SOHBET (Çıkış: q)")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n❓ Siz: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGörüşürüz! 👋")
            break
            
        if user_input.lower() in ('q', 'quit', 'exit'):
            print("Görüşürüz! 👋")
            break
        
        if not user_input:
            continue
        
        full_prompt = format_sft_prompt(user_input)
        
        generated = generate_text(
            model, tokenizer, device,
            full_prompt,
            max_new_tokens=150,
            temperature=0.7,
            top_k=50,
            repetition_penalty=1.2
        )
        
        answer = clean_sft_output(generated)
        print(f"🤖 Luna: {answer}")


if __name__ == "__main__":
    main()
