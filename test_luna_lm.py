"""
Luna-LM Test Script
En son eğitilen modeli test etmek için
"""

import torch
import json
import os

from model import GPTModel, generate_text
from turkish_tokenizer_pretrained import PretrainedTurkishTokenizer


def load_model(checkpoint_path, device='cuda'):
    """Model ve tokenizer'ı yükle. checkpoint_path bir klasör VEYA .pt dosyası olabilir."""
    
    model_config = None
    tokenizer_name = 'dbmdz/bert-base-turkish-cased'  # Varsayılan
    
    # Durum 1: checkpoint_path bir KLASÖR (Pretraining çıktısı)
    if os.path.isdir(checkpoint_path):
        config_path = os.path.join(checkpoint_path, "config.json")
        weights_path = os.path.join(checkpoint_path, "best_model.pt")
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            model_config = config['model_config']
            tokenizer_name = config.get('tokenizer', tokenizer_name)
    
    # Durum 2: checkpoint_path bir DOSYA (.pt - SFT çıktısı)
    elif os.path.isfile(checkpoint_path):
        weights_path = checkpoint_path
        
        base_dir = os.path.dirname(checkpoint_path) or '.'
        config_path = os.path.join(base_dir, "config.json")
        if os.path.exists(config_path):
             with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                model_config = config['model_config']
        else:
             print("UYARI: Config dosyası bulunamadı, varsayılan 'small' config kullanılıyor.")
             model_config = {
                "vocab_size": 32000,
                "context_length": 512,
                "emb_dim": 512,
                "n_heads": 8,
                "n_layers": 6,
                "drop_rate": 0.1,
                "qkv_bias": False
            }
    else:
        raise ValueError(f"Geçersiz yol: {checkpoint_path}")

    tokenizer = PretrainedTurkishTokenizer(tokenizer_name)
    model_config['vocab_size'] = tokenizer.vocab_size

    print(f"Model config: {model_config}")
    
    model = GPTModel(model_config)
    
    print(f"Ağırlıklar yükleniyor: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, tokenizer, model_config


def main():
    print("="*60)
    print("LUNA-LM TEST")
    print("="*60)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Model yükle
    # Önce SFT modeline bak, yoksa klasöre bak
    if os.path.exists("luna_sft_finetuned.pt"):
        checkpoint_path = "luna_sft_finetuned.pt"
    else:
        checkpoint_path = "luna_lm_checkpoints_20251218_121142"
        
    print(f"\n📦 Model Yolu: {checkpoint_path}")
    
    try:
        model, tokenizer, config = load_model(checkpoint_path, device)
    except Exception as e:
        print(f"\n⚠️ Model yüklenirken hata oluştu: {e}")
        print(f"Lütfen 'luna_sft_finetuned.pt' dosyasının bu klasörde olduğundan emin olun.")
        return
    
    # SFT Prompt Format
    SYSTEM_PROMPT = "Senin adın Luna. Amacın insanlara yardımcı olmak ve sorulara açık, anlaşılır cevaplar vermektir. Emin olmadığın konularda bunu belirtir, uydurma bilgi eklemezsin. Cevaplarını nazik, sade ve doğal bir Türkçe ile yazarsın."
    
    def format_sft_prompt(user_query):
        return f"<system>{SYSTEM_PROMPT}</system>\n<user>{user_query}</user>\n<assistant>"

    # Test prompts (SFT için Soru Formatında)
    print("\n" + "="*60)
    print("SFT FORMATLI METİN ÜRETİMİ TESTİ")
    print("="*60)
    
    test_questions = [
        "Güneş hangi yönden doğar?", # Dataset'te VAR
        "Ampulü kim buldu?",         # Dataset'te VAR
        "İstanbul'un önemi nedir?",  # Dataset'te YOK (Benzeri var ama aynısı değil)
        "Mutluluk nedir?",           # Dataset'te VAR
        "Bana bir hikaye anlat.",    # Dataset'te VAR
        "Kravat nasıl bağlanır?",    # Dataset'te VAR
    ]
    
    for q in test_questions:
        print(f"\n❓ Soru: '{q}'")
        print("-" * 40)
        
        full_prompt = format_sft_prompt(q)
        
        # Temperature'ı düşürdüm (0.2). Model küçük olduğu için yaratıcılık = saçmalama oluyor.
        generated = generate_text(
            model, tokenizer, device,
            full_prompt, 
            max_new_tokens=100, 
            temperature=0.2,    
            top_k=40,
            repetition_penalty=1.2 
        )
        
        # === AKILLI TEMİZLİK (Tokenizer artifactlerini temizle) ===
        # Tokenizer < system > şeklinde boşluklu üretebiliyor, bunları normalleştirelim
        clean_gen = generated.replace("< assistant >", "<assistant>").replace("< / assistant >", "</assistant>")
        clean_gen = clean_gen.replace("< user >", "<user>").replace("< / user >", "</user>")
        clean_gen = clean_gen.replace("< system >", "<system>").replace("< / system >", "</system>")
        
        # En son açılan <assistant> taginden sonrasını al (Cevap oradadır)
        if "<assistant>" in clean_gen:
             answer = clean_gen.split("<assistant>")[-1]
        else:
             answer = clean_gen
            
        # Stop token kontrolü (Cevabın bittiği yer)
        for stop_token in ["</assistant>", "<user>", "<system>"]:
            if stop_token in answer:
                answer = answer.split(stop_token)[0]
        
        # Gereksiz karakter temizliği
        answer = answer.strip()
        while answer.startswith(">") or answer.startswith(" ") or answer.startswith("."):
            answer = answer[1:].strip()
            
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
        
        # === AYNI TEMİZLİK MANTIĞI ===
        clean_gen = generated.replace("< assistant >", "<assistant>").replace("< / assistant >", "</assistant>")
        clean_gen = clean_gen.replace("< user >", "<user>").replace("< / user >", "</user>")
        clean_gen = clean_gen.replace("< system >", "<system>").replace("< / system >", "</system>")
        
        if "<assistant>" in clean_gen:
             answer = clean_gen.split("<assistant>")[-1]
        else:
             answer = clean_gen
            
        for stop_token in ["</assistant>", "<user>", "<system>"]:
            if stop_token in answer:
                answer = answer.split(stop_token)[0]
        
        answer = answer.strip()
        while answer.startswith(">") or answer.startswith(" ") or answer.startswith("."):
             answer = answer[1:].strip()

        print(f"Luna: {answer}")


if __name__ == "__main__":
    main()
