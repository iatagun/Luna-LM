"""
Luna-LM SFT (Supervised Fine-Tuning) Script
Colab Uyumlu - Tek Dosya Hali
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
import random

from model import GPTModel
from turkish_tokenizer_pretrained import PretrainedTurkishTokenizer

class SFTDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.data.append(item['text'])
        
        print(f"📖 SFT Dataset yüklendi: {len(self.data)} örnek")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        # Encode
        token_ids = self.tokenizer.encode(text)
        
        # Truncate / Padding (Basit sliding window)
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        
        input_tensor = torch.tensor(token_ids, dtype=torch.long)
        
        # GPT Training: input=x, target=x shifted right
        # PyTorch CrossEntropyLoss handles the shift internally via indexing during calc
        # But here we standardly return input sequence for causal modeling
        return input_tensor, input_tensor

def collate_fn(batch):
    # Dynamic padding for batch
    max_len = max(len(x[0]) for x in batch)
    
    # Pad inputs with 0 (or pad token if available, using 0 for simplicity here as pad often is 0 in BERT)
    padded_inputs = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_targets = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    for i, (inp, tgt) in enumerate(batch):
        l = len(inp)
        padded_inputs[i, :l] = inp
        padded_targets[i, :l] = tgt
        
    return padded_inputs, padded_targets

# ==================== 3. TRAINING LOOP ====================

def train_sft(model, dataloader, optimizer, device, epochs, pad_token_id):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (inp, tgt) in enumerate(dataloader):
            inp, tgt = inp.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            logits = model(inp)
            
            # Shift for loss calculation
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = tgt[:, 1:].contiguous()
            
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1),
                ignore_index=pad_token_id # DÜZELTİLDİ: Dinamik pad_token_id
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # EKLENDİ: Grad Clip
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} Completed | Avg Loss: {avg_loss:.4f}")

    return model

# ==================== 4. MAIN EXECUTION ====================

def main():
    print("🚀 SFT (Fine-Tuning) Başlatılıyor...")
    
    # AYARLAR
    # Modeli direkt bulunduğumuz dizinden (root) arar
    DATASET_FILE = "sft_dataset_luna_text.jsonl"
    CONFIG_FILE = "config.json"
    WEIGHTS_FILE = "best_model.pt"
    
    EPOCHS = 10       # DÜZELTİLDİ: Küçük model + Büyük dataset için daha fazla tekrar
    LR = 3e-5         # DÜZELTİLDİ: Hassas öğrenme
    BATCH_SIZE = 8    # DÜZELTİLDİ: Daha stabil gradient
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    # 1. Config Yükle
    if not os.path.exists(CONFIG_FILE):
        print(f"HATA: {CONFIG_FILE} bulunamadı! Lütfen Colab'de dosyaları (config.json, best_model.pt) direkt ana dizine yükleyin.")
        return

    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    model_config = config['model_config']
    
    # 2. Tokenizer
    tokenizer = PretrainedTurkishTokenizer('dbmdz/bert-base-turkish-cased')
    pad_id = tokenizer.tokenizer.pad_token_id # Pad ID al
    
    # 3. Model Yükle
    model = GPTModel(model_config)
    
    if os.path.exists(WEIGHTS_FILE):
        print(f"Model yükleniyor: {WEIGHTS_FILE}")
        checkpoint = torch.load(WEIGHTS_FILE, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Pretrained ağırlıklar yüklendi.")
    else:
        print(f"HATA: {WEIGHTS_FILE} bulunamadı! Lütfen Colab'de dosyaları direkt ana dizine yükleyin.")
        return
        
    # --- OPTİMİZASYON: Embedding Dondurma (SFT 1. Tur için) ---
    print("❄️ Embedding katmanları donduruluyor (Stabilite için)...")
    for p in model.tok_emb.parameters():
        p.requires_grad = False
    for p in model.pos_emb.parameters():
        p.requires_grad = False
        
    model.to(DEVICE)
    
    # 4. Dataset Hazırla
    if not os.path.exists(DATASET_FILE):
        print(f"HATA: {DATASET_FILE} yok! Lütfen colab'e yükleyin.")
        return

    dataset = SFTDataset(DATASET_FILE, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # 5. Optimizer (Weight Decay Eklendi)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    # 6. Eğitim
    model = train_sft(model, dataloader, optimizer, DEVICE, EPOCHS, pad_id)
    
    # 7. Kaydet
    save_path = "luna_sft_finetuned.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\n🎉 SFT Tamamlandı! Model kaydedildi: {save_path}")
    print("Bu dosyayı bilgisayarınıza indirin.")

    # 8. Hızlı Test
    print("\n🔍 Test ediliyor...")
    model.eval()
    test_q = "<system>Senin adın Luna. Amacın insanlara yardımcı olmak ve sorulara açık, anlaşılır cevaplar vermektir. Emin olmadığın konularda bunu belirtir, uydurma bilgi eklemezsin. Cevaplarını nazik, sade ve doğal bir Türkçe ile yazarsın.</system>\n<user>Güneş neden parlar?</user>\n<assistant>"
    
    encoded = torch.tensor(tokenizer.encode(test_q)).unsqueeze(0).to(DEVICE)
    out = model(encoded) # Logits
    # (Burada basit generation fonksiyonu olmadığı için sadece çalıştığını teyit ediyoruz)
    print("Test generation (logits calculated) - Başarılı.")

if __name__ == "__main__":
    main()
