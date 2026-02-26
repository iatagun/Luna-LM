"""
Luna-LM SFT (Supervised Fine-Tuning) Eğitim Scripti
Pretrained model üzerine instruction-following yeteneği kazandırır.

83K+ satırlık büyük SFT dataset desteği.
Referans: LLMs-from-scratch ch07 (Sebastian Raschka)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import random
import time
import matplotlib.pyplot as plt
from datetime import datetime

from luna.model import GPTModel
from luna.tokenizer import PretrainedTurkishTokenizer
from luna.generate import generate_text
from luna.utils import load_model


# ==================== PROMPT FORMAT ====================

SYSTEM_PROMPT = (
    "Senin adın Luna. Amacın insanlara yardımcı olmak ve sorulara açık, "
    "anlaşılır cevaplar vermektir. Emin olmadığın konularda bunu belirtir, "
    "uydurma bilgi eklemezsin. Cevaplarını nazik, sade ve doğal bir Türkçe ile yazarsın."
)


def format_input(entry):
    """Instruction + input formatla (system/user/assistant)"""
    return (
        f"<system>{SYSTEM_PROMPT}</system>\n"
        f"<user>{entry['user']}</user>\n"
        f"<assistant>"
    )


# ==================== SFT DATASET ====================

class SFTDataset(Dataset):
    """
    Büyük SFT datasetleri için memory-efficient Dataset.
    Pre-tokenize eder, progress gösterir.
    """
    
    def __init__(self, data, tokenizer, max_length=512):
        self.encoded_texts = []
        
        too_long = 0
        too_short = 0
        
        for i, entry in enumerate(data):
            if (i + 1) % 10000 == 0:
                print(f"    Tokenize: {i+1:,}/{len(data):,}...")
            
            instruction_part = format_input(entry)
            response_part = f"{entry['assistant']}</assistant>"
            full_text = instruction_part + response_part
            
            tokens = tokenizer.encode(full_text)
            
            # Çok kısa örnekleri atla
            if len(tokens) < 10:
                too_short += 1
                continue
            
            # Context length'i aşan örnekleri ATLA (truncate değil!)
            # Truncate edersek cevaplar yarım kalır ve alignment bozulur
            if len(tokens) > max_length:
                too_long += 1
                continue
            
            self.encoded_texts.append(tokens)
        
        total_skipped = too_long + too_short
        print(f"  ✓ SFTDataset: {len(self.encoded_texts):,} sample kullanılacak")
        print(f"    Atlanan (>{max_length} token): {too_long:,}")
        print(f"    Atlanan (<10 token):          {too_short:,}")
        print(f"    Toplam atlanan:               {total_skipped:,} / {len(data):,} ({100*total_skipped/max(len(data),1):.1f}%)")
    
    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    def __len__(self):
        return len(self.encoded_texts)


def sft_collate_fn(batch, pad_token_id=0, ignore_index=-100,
                   allowed_max_length=None, device="cpu"):
    """
    ch07 referansına uygun collate fonksiyonu.
    
    1. Her örneğin sonuna EOS (pad_token_id) ekle
    2. Batch'teki en uzun örneğe göre pad'le
    3. input = padded[:-1], target = padded[1:]
    4. Target'ta İLK pad token (EOS) tutuluyor → model cevabın bittiğini öğrenir
    5. Geri kalan pad tokenlar -100 ile maskeleniyor → loss'a katılmaz
    """
    batch_max_length = max(len(item) + 1 for item in batch)
    
    inputs_lst, targets_lst = [], []
    
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        
        # İLK pad (EOS) tut, GERİ KALANLARINI maskele
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    
    return inputs_tensor, targets_tensor


# ==================== LOSS & EVAL ====================

def calc_loss_batch(input_batch, target_batch, model, device):
    """Tek batch için loss hesapla"""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """DataLoader üzerinde ortalama loss"""
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    
    return total_loss / num_batches


def plot_losses(train_losses, val_losses, tokens_seen, save_path):
    """Train/Val loss grafiğini çiz ve kaydet"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    steps = range(1, len(train_losses) + 1)
    ax1.plot(steps, train_losses, label="Training loss", linewidth=2)
    ax1.plot(steps, val_losses, linestyle="-.", label="Validation loss", linewidth=2)
    ax1.set_xlabel("Eval Steps")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Luna-LM SFT Training Progress")
    
    # İkinci x-axis: tokens seen
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens Seen")
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  ✓ Loss grafiği kaydedildi: {save_path}")


# ==================== SFT EĞİTİM ====================

def train_sft(model, train_loader, val_loader, optimizer, scheduler, device,
              num_epochs, save_dir, tokenizer=None, eval_freq=5, eval_iter=5,
              start_context=None):
    """
    SFT eğitim loop'u — ch07 referansına uygun + Cosine LR Scheduler.
    """
    
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1
    best_val_loss = float('inf')
    
    total_steps = len(train_loader) * num_epochs
    
    print(f"\n{'='*60}")
    print("SFT EĞİTİM BAŞLIYOR")
    print(f"{'='*60}")
    print(f"Epochs: {num_epochs}")
    print(f"Train batches/epoch: {len(train_loader):,}")
    print(f"Total steps: {total_steps:,}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Eval freq: her {eval_freq} step")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.time()
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            tokens_seen += input_batch.numel()
            global_step += 1
            
            # Evaluation
            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
                    val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Ep {epoch+1} | Step {global_step:,} | "
                      f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                      f"LR: {current_lr:.2e} | Tokens: {tokens_seen:,}")
                
                # Best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = os.path.join(save_dir, "best_sft_model.pt")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': epoch,
                        'global_step': global_step,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'tokens_seen': tokens_seen,
                    }, save_path)
                    print(f"  ✓ Best model! Val: {val_loss:.4f}")
                
                model.train()
        
        # Epoch sonu
        epoch_time = time.time() - epoch_start
        print(f"\n{'='*40}")
        print(f"Epoch {epoch+1}/{num_epochs} tamamlandı ({epoch_time/60:.1f} dk)")
        print(f"{'='*40}")
        
        # Örnek üretim
        if tokenizer is not None and start_context is not None:
            model.eval()
            generated = generate_text(
                model, tokenizer, device, start_context,
                max_new_tokens=80, temperature=0.3, top_k=40,
                repetition_penalty=1.2
            )
            if "<assistant>" in generated:
                answer = generated.split("<assistant>")[-1]
                for stop in ["</assistant>", "<user>", "<system>"]:
                    if stop in answer:
                        answer = answer.split(stop)[0]
                answer = answer.strip()
            else:
                answer = generated
            print(f"  📝 Örnek: {answer}")
            model.train()
        
        # Epoch checkpoint
        save_path = os.path.join(save_dir, f"sft_epoch_{epoch+1}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'tokens_seen': tokens_seen,
        }, save_path)
        print()
    
    print(f"\nSFT Eğitim tamamlandı! Best val loss: {best_val_loss:.4f}")
    return train_losses, val_losses, track_tokens_seen


# ==================== VERİ YÜKLEME ====================

def load_sft_data(jsonl_path):
    """JSONL dosyasından SFT verisi yükle (memory-efficient, satır satır)"""
    
    print(f"  Dosya: {jsonl_path}")
    data = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # user ve assistant alanları zorunlu
                if 'user' in entry and 'assistant' in entry:
                    data.append(entry)
                # Alternatif alan adları desteği
                elif 'instruction' in entry and 'output' in entry:
                    data.append({
                        'user': entry['instruction'] + (f"\n{entry['input']}" if entry.get('input') else ''),
                        'assistant': entry['output']
                    })
                elif 'question' in entry and 'answer' in entry:
                    data.append({
                        'user': entry['question'],
                        'assistant': entry['answer']
                    })
            except json.JSONDecodeError:
                continue
            
            if (i + 1) % 20000 == 0:
                print(f"    Okunan: {i+1:,} satır...")
    
    print(f"  ✓ Toplam geçerli örnek: {len(data):,}")
    return data


# ==================== MAIN ====================

def main():
    print("\n" + "="*60)
    print("LUNA-LM SFT EĞİTİMİ (83K+ Dataset)")
    print("="*60 + "\n")
    
    # ==========================================
    # Ayarlar — 83K dataset için optimize
    # ==========================================
    BATCH_SIZE = 8          # VRAM'e göre: 4 (8GB), 8 (16GB), 16 (24GB+)
    NUM_EPOCHS = 2          # 83K * 2 = 166K step yeterli
    LEARNING_RATE = 5e-5    # ch07 referansı
    WEIGHT_DECAY = 0.1      # ch07 referansı
    EVAL_FREQ = 200         # 83K dataset → daha seyrek eval
    EVAL_ITER = 10          # Loss tahmininde kullanılacak batch sayısı
    MAX_LENGTH = 512        # Context length
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    project_root = os.path.join(os.path.dirname(__file__), '..')
    
    # ==========================================
    # 1. Pretrained model yükle
    # ==========================================
    print("\n1. Pretrained model yükleniyor...")
    
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
    
    pad_token_id = tokenizer.tokenizer.sep_token_id or 0
    context_length = model_config.get("context_length", MAX_LENGTH)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params/1e6:.1f}M parametre")
    print(f"  Pad token ID: {pad_token_id}")
    print(f"  Context length: {context_length}")
    
    # ==========================================
    # 2. SFT veri yükle ve split et
    # ==========================================
    print("\n2. SFT dataset yükleniyor...")
    
    # Veri dosyasını bul
    sft_data_path = None
    search_paths = [
        os.path.join(project_root, "sft", "sft_dataset.jsonl"),
        os.path.join(project_root, "sft_dataset_luna_text.jsonl"),
        os.path.join(project_root, "sft_dataset.jsonl"),
    ]
    for p in search_paths:
        if os.path.exists(p):
            sft_data_path = p
            break
    
    if sft_data_path is None:
        print("❌ SFT dataset bulunamadı! Beklenen konumlar:")
        for p in search_paths:
            print(f"   - {p}")
        return
    
    all_data = load_sft_data(sft_data_path)
    
    if len(all_data) == 0:
        print("❌ Dataset boş!")
        return
    
    # Train/Val/Test split (85/10/5) — ch07 referansı
    random.seed(42)
    random.shuffle(all_data)
    
    train_portion = int(len(all_data) * 0.85)
    test_portion = int(len(all_data) * 0.10)
    
    train_data = all_data[:train_portion]
    test_data = all_data[train_portion:train_portion + test_portion]
    val_data = all_data[train_portion + test_portion:]
    
    del all_data  # Bellek temizle
    
    print(f"\n  Split:")
    print(f"    Train: {len(train_data):,}")
    print(f"    Val:   {len(val_data):,}")
    print(f"    Test:  {len(test_data):,}")
    
    # ==========================================
    # 3. Dataset & DataLoader
    # ==========================================
    print("\n3. Tokenize & DataLoader oluşturuluyor...")
    
    train_dataset = SFTDataset(train_data, tokenizer, max_length=context_length)
    val_dataset = SFTDataset(val_data, tokenizer, max_length=context_length)
    
    customized_collate_fn = partial(
        sft_collate_fn,
        pad_token_id=pad_token_id,
        ignore_index=-100,
        allowed_max_length=context_length,
        device=device
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    
    print(f"  Train: {len(train_loader):,} batch")
    print(f"  Val:   {len(val_loader):,} batch")
    
    # ==========================================
    # 4. Optimizer & Cosine LR Scheduler
    # ==========================================
    print("\n4. Optimizer & Scheduler ayarlanıyor...")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    total_training_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_training_steps * 0.05)  # %5 warmup
    
    # Cosine Annealing with Warmup (linear warmup → cosine decay)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)  # Linear warmup
        progress = (step - warmup_steps) / max(total_training_steps - warmup_steps, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"  ✓ AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
    print(f"  ✓ Cosine scheduler ({total_training_steps:,} steps, {warmup_steps} warmup)")
    
    # ==========================================
    # 5. Save directory
    # ==========================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(project_root, f"checkpoints/sft_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    config_save = {
        'model_config': model_config,
        'sft_config': {
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'max_length': MAX_LENGTH,
            'eval_freq': EVAL_FREQ,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'total_steps': total_training_steps,
            'warmup_steps': warmup_steps,
        },
        'tokenizer': 'dbmdz/bert-base-turkish-cased',
        'base_checkpoint': str(checkpoint_dir),
        'timestamp': timestamp,
    }
    
    with open(os.path.join(save_dir, "config.json"), 'w', encoding='utf-8') as f:
        json.dump(config_save, f, indent=2, ensure_ascii=False)
    
    # Test verisi kaydet (sonra eval için)
    with open(os.path.join(save_dir, "test_data.json"), 'w', encoding='utf-8') as f:
        json.dump(test_data[:100], f, indent=2, ensure_ascii=False)  # İlk 100 test
    
    print(f"\n5. Save dir: {save_dir}")
    
    # ==========================================
    # 6. Eğitim!
    # ==========================================
    start_context = format_input(val_data[0]) if val_data else None
    
    start_time = time.time()
    
    train_losses, val_losses, tokens_seen = train_sft(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=NUM_EPOCHS,
        save_dir=save_dir,
        tokenizer=tokenizer,
        eval_freq=EVAL_FREQ,
        eval_iter=EVAL_ITER,
        start_context=start_context,
    )
    
    total_time = (time.time() - start_time) / 60
    print(f"\nToplam eğitim süresi: {total_time:.1f} dakika")
    
    # ==========================================
    # 7. Loss grafiği
    # ==========================================
    print("\n7. Loss grafiği kaydediliyor...")
    plot_path = os.path.join(save_dir, "sft_loss.png")
    plot_losses(train_losses, val_losses, tokens_seen, plot_path)
    
    # ==========================================
    # 8. Test set üzerinde response üret
    # ==========================================
    print("\n" + "="*60)
    print("8. TEST SET DEĞERLENDİRME (ilk 5 örnek)")
    print("="*60)
    
    model.eval()
    for i, entry in enumerate(test_data[:5]):
        input_text = format_input(entry)
        
        generated = generate_text(
            model=model, tokenizer=tokenizer, device=device,
            start_text=input_text, max_new_tokens=100,
            temperature=0.3, top_k=40, repetition_penalty=1.2
        )
        
        if "<assistant>" in generated:
            response = generated.split("<assistant>")[-1]
            for stop in ["</assistant>", "<user>", "<system>"]:
                if stop in response:
                    response = response.split(stop)[0]
            response = response.strip()
        else:
            response = generated
        
        print(f"\n❓ Soru: {entry['user']}")
        print(f"📗 Beklenen: {entry['assistant'][:100]}...")
        print(f"🤖 Model:   {response[:100]}...")
    
    # ==========================================
    # Özet
    # ==========================================
    print("\n" + "="*60)
    print("SFT EĞİTİM TAMAMLANDI! 🎉")
    print("="*60)
    print(f"\n✓ Checkpoints: {save_dir}/")
    print(f"✓ En iyi model: {save_dir}/best_sft_model.pt")
    print(f"✓ Loss grafiği: {save_dir}/sft_loss.png")
    print(f"✓ Config: {save_dir}/config.json")
    print(f"\n📊 Eğitim özeti:")
    print(f"   Süre: {total_time:.1f} dakika")
    print(f"   Final train loss: {train_losses[-1]:.4f}" if train_losses else "")
    print(f"   Final val loss: {val_losses[-1]:.4f}" if val_losses else "")
    print(f"\nTest etmek için:")
    print(f"  python scripts/test_model.py")


if __name__ == "__main__":
    main()
