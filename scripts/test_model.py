"""
Luna-LM Test Script
Eğitilmiş modeli (pretrained veya SFT) test etmek için
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

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
    for stop_token in ["</assistant>", "<user>", "<system>"]:
        if stop_token in answer:
            answer = answer.split(stop_token)[0]
    
    # Gereksiz karakter temizliği
    answer = answer.strip()
    while answer and (answer[0] in ('>', ' ', '.')):
        answer = answer[1:].strip()
        
    return answer


def main():
    print("="*60)
    print("LUNA-LM TEST")
    print("="*60)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Proje kök dizini
    project_root = os.path.join(os.path.dirname(__file__), '..')
    
    # Model yükle — önce SFT modeline bak, yoksa checkpoint klasörüne
    sft_path = os.path.join(project_root, "luna_sft_finetuned.pt")
    pretrain_path = os.path.join(project_root, "luna_lm_checkpoints_20251218_121142")
    
    if os.path.exists(sft_path):
        checkpoint_path = sft_path
    elif os.path.exists(pretrain_path):
        checkpoint_path = pretrain_path
    else:
        print("❌ Model bulunamadı!")
        return
        
    print(f"\n📦 Model Yolu: {checkpoint_path}")
    
    try:
        model, tokenizer, config = load_model(checkpoint_path, device)
    except Exception as e:
        print(f"\n⚠️ Model yüklenirken hata oluştu: {e}")
        return
    
    # Test prompts (SFT için Soru Formatında)
    print("\n" + "="*60)
    print("SFT FORMATLI METİN ÜRETİMİ TESTİ")
    print("="*60)
    
    test_questions = [
        "Güneş hangi yönden doğar?",
        "Ampulü kim buldu?",
        "İstanbul'un önemi nedir?",
        "Mutluluk nedir?",
        "Bana bir hikaye anlat.",
        "Kravat nasıl bağlanır?",
    ]
    
    for q in test_questions:
        print(f"\n❓ Soru: '{q}'")
        print("-" * 40)
        
        full_prompt = format_sft_prompt(q)
        
        generated = generate_text(
            model, tokenizer, device,
            full_prompt, 
            max_new_tokens=100, 
            temperature=0.2,    
            top_k=40,
            repetition_penalty=1.2 
        )
        
        answer = clean_sft_output(generated)
        print(f"🤖 Luna: {answer}")

    # İnteraktif mod
    print("\n" + "="*60)
    print("İNTERAKTİF SOHBET MODU (Çıkış için 'q')")
    print("="*60)
    
    while True:
        user_input = input("\nSiz: ").strip()
        if user_input.lower() == 'q':
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
        print(f"Luna: {answer}")


if __name__ == "__main__":
    main()
