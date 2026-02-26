"""
Luna-LM SFT (Supervised Fine-Tuning) Eğitim Scripti
Pretrained model üzerine instruction-following yeteneği kazandırır.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import random
from datetime import datetime

from luna.model import GPTModel, MODEL_CONFIGS
from luna.tokenizer import PretrainedTurkishTokenizer
from luna.generate import generate_text
from luna.utils import load_model


# ==================== SFT DATASET ====================

SYSTEM_PROMPT = "Senin adın Luna. Amacın insanlara yardımcı olmak ve sorulara açık, anlaşılır cevaplar vermektir. Emin olmadığın konularda bunu belirtir, uydurma bilgi eklemezsin. Cevaplarını nazik, sade ve doğal bir Türkçe ile yazarsın."


class SFTDataset(Dataset):
    """SFT formatındaki JSONL dosyasını okuyan Dataset"""
    
    def __init__(self, jsonl_path, tokenizer, max_length=512, system_prompt=SYSTEM_PROMPT):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                
                # Format: <system>...<user>...<assistant>...</assistant>
                formatted = (
                    f"<system>{system_prompt}</system>\n"
                    f"<user>{data['user']}</user>\n"
                    f"<assistant>{data['assistant']}</assistant>"
                )
                
                token_ids = tokenizer.encode(formatted)
                
                # Truncate
                if len(token_ids) > max_length:
                    token_ids = token_ids[:max_length]
                
                if len(token_ids) >= 2:
                    self.samples.append(token_ids)
        
        print(f"SFT Dataset: {len(self.samples)} sample yüklendi ({jsonl_path})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        token_ids = self.samples[idx]
        input_ids = torch.tensor(token_ids[:-1])
        target_ids = torch.tensor(token_ids[1:])
        return input_ids, target_ids


def sft_collate_fn(batch):
    """Değişken uzunluklu örnekleri pad'le"""
    input_ids_list, target_ids_list = zip(*batch)
    
    max_len = max(x.size(0) for x in input_ids_list)
    
    padded_inputs = []
    padded_targets = []
    
    for inp, tgt in zip(input_ids_list, target_ids_list):
        pad_len = max_len - inp.size(0)
        if pad_len > 0:
            inp = torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)])
            tgt = torch.cat([tgt, torch.full((pad_len,), -100, dtype=torch.long)])  # -100 = ignore
        padded_inputs.append(inp)
        padded_targets.append(tgt)
    
    return torch.stack(padded_inputs), torch.stack(padded_targets)


# ==================== SFT EĞİTİM ====================

def train_sft(model, train_loader, optimizer, device, num_epochs, save_dir,
              tokenizer=None, eval_every=50):
    """SFT eğitim loop'u"""
    
    model.train()
    global_step = 0
    best_loss = float('inf')
    
    print(f"\n{'='*60}")
    print("SFT EĞİTİM BAŞLIYOR")
    print(f"{'='*60}")
    print(f"Epochs: {num_epochs}")
    print(f"Batches: {len(train_loader)}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids)
            
            # Cross entropy loss (ignore_index=-100 ile pad tokenları atlanır)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=-100
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            if global_step % eval_every == 0:
                avg_loss = epoch_loss / num_batches
                print(f"Epoch {epoch+1}/{num_epochs} | Step {global_step} | Loss: {avg_loss:.4f}")
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_path = os.path.join(save_dir, "best_sft_model.pt")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': epoch,
                        'global_step': global_step,
                        'loss': avg_loss,
                    }, save_path)
                    print(f"  ✓ En iyi SFT model kaydedildi! Loss: {avg_loss:.4f}")
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"\n{'='*40}")
        print(f"Epoch {epoch+1}/{num_epochs} Tamamlandı | Loss: {avg_epoch_loss:.4f}")
        print(f"{'='*40}\n")
        
        # Epoch checkpoint
        save_path = os.path.join(save_dir, f"sft_epoch_{epoch+1}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'loss': avg_epoch_loss,
        }, save_path)
        
        # Örnek üretim
        if tokenizer is not None:
            model.eval()
            test_prompt = f"<system>{SYSTEM_PROMPT}</system>\n<user>Merhaba, nasılsın?</user>\n<assistant>"
            generated = generate_text(
                model, tokenizer, device, test_prompt,
                max_new_tokens=80, temperature=0.3, top_k=40
            )
            print(f"  📝 Örnek: {generated}\n")
            model.train()
    
    print(f"\nSFT Eğitim tamamlandı! Best loss: {best_loss:.4f}")
    return best_loss


# ==================== MAIN ====================

def main():
    print("\n" + "="*60)
    print("LUNA-LM SFT EĞİTİMİ")
    print("="*60 + "\n")
    
    # Ayarlar
    BATCH_SIZE = 4
    NUM_EPOCHS = 10
    LEARNING_RATE = 5e-5
    MAX_LENGTH = 512
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    project_root = os.path.join(os.path.dirname(__file__), '..')
    
    # 1. Pretrained model yükle
    print("\n1. Pretrained model yükleniyor...")
    
    # Checkpoint bul (önce yeni yapı, sonra eski)
    import glob
    checkpoint_dirs = glob.glob(os.path.join(project_root, "checkpoints", "pretrain_*"))
    if not checkpoint_dirs:
        checkpoint_dirs = glob.glob(os.path.join(project_root, "luna_lm_checkpoints_*"))
    
    if not checkpoint_dirs:
        print("❌ Pretrained model bulunamadı! Önce scripts/train.py çalıştırın.")
        return
    
    checkpoint_dir = sorted(checkpoint_dirs)[-1]
    print(f"  Checkpoint: {checkpoint_dir}")
    
    model, tokenizer, model_config = load_model(checkpoint_dir, device=device)
    model.train()
    
    # 2. SFT veri yükle
    print("\n2. SFT dataset yükleniyor...")
    
    sft_data_path = os.path.join(project_root, "sft", "sft_dataset.jsonl")
    if not os.path.exists(sft_data_path):
        # Eski konuma bak
        sft_data_path = os.path.join(project_root, "sft_dataset_luna_text.jsonl")
    
    if not os.path.exists(sft_data_path):
        print(f"❌ SFT dataset bulunamadı!")
        print(f"   Beklenen: sft/sft_dataset.jsonl")
        print(f"   Önce sft/generate_sft_data.py çalıştırın.")
        return
    
    dataset = SFTDataset(sft_data_path, tokenizer, max_length=MAX_LENGTH)
    
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=sft_collate_fn,
        drop_last=True
    )
    
    print(f"  ✓ {len(dataset)} sample, {len(train_loader)} batch")
    
    # 3. Optimizer
    print("\n3. Optimizer ayarlanıyor...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    print(f"  ✓ AdamW (lr={LEARNING_RATE})")
    
    # 4. Save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(project_root, f"checkpoints/sft_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Config kaydet
    with open(os.path.join(save_dir, "config.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'model_config': model_config,
            'sft_config': {
                'batch_size': BATCH_SIZE,
                'num_epochs': NUM_EPOCHS,
                'learning_rate': LEARNING_RATE,
                'max_length': MAX_LENGTH,
            },
            'tokenizer': 'dbmdz/bert-base-turkish-cased',
            'base_checkpoint': str(checkpoint_dir),
            'timestamp': timestamp,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"  Save dir: {save_dir}")
    
    # 5. Eğitim!
    train_sft(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        save_dir=save_dir,
        tokenizer=tokenizer,
    )
    
    print("\n" + "="*60)
    print("SFT EĞİTİM TAMAMLANDI! 🎉")
    print("="*60)
    print(f"\n✓ Checkpoints: {save_dir}/")
    print(f"✓ En iyi model: {save_dir}/best_sft_model.pt")
    print(f"\nTest etmek için:")
    print(f"  python scripts/test_model.py")


if __name__ == "__main__":
    main()
