"""
Luna-LM Foundation Model Eğitimi
foundation_corpus_clean.txt ile Türkçe dil modeli pretraining
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Luna paketi
from luna.tokenizer import PretrainedTurkishTokenizer
from luna.data import create_dataloader_pretrained
from luna.model import GPTModel, MODEL_CONFIGS, get_model_config
from luna.generate import generate_text


# ==================== EĞİTİM FONKSİYONLARI ====================

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


def calc_loss_and_accuracy(data_loader, model, device, num_batches=None):
    """DataLoader üzerinde ortalama loss ve accuracy hesapla"""
    total_loss = 0.
    total_correct = 0
    total_tokens = 0
    
    if len(data_loader) == 0:
        return float("nan"), 0.0
    
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), 
            target_batch.flatten()
        )
        total_loss += loss.item()
        
        # Accuracy hesapla
        predictions = logits.argmax(dim=-1)
        correct = (predictions == target_batch).sum().item()
        total_correct += correct
        total_tokens += target_batch.numel()
    
    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, accuracy


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """Model değerlendirme — loss ve accuracy"""
    model.eval()
    with torch.no_grad():
        train_loss, train_acc = calc_loss_and_accuracy(train_loader, model, device, num_batches=eval_iter)
        val_loss, val_acc = calc_loss_and_accuracy(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, train_acc, val_loss, val_acc


def train_model(model, train_loader, val_loader, optimizer, device, 
                num_epochs, eval_freq, eval_iter, save_dir, tokenizer,
                start_context="Bugün", scheduler=None):
    """Ana eğitim loop'u"""
    
    # Training tracking
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = 0
    best_val_loss = float('inf')
    
    print(f"\n{'='*60}")
    print("EĞİTİM BAŞLIYOR")
    print(f"{'='*60}")
    print(f"Toplam epoch: {num_epochs}")
    print(f"Batch sayısı: {len(train_loader)}")
    print(f"Değerlendirme sıklığı: Her {eval_freq} step")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            tokens_seen += input_batch.numel()
            global_step += 1
            epoch_loss += loss.item()
            
            # Evaluation
            if global_step % eval_freq == 0 and global_step > 0:
                print()
                train_loss, train_acc, val_loss, val_acc = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{num_epochs} | Step {global_step:,} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | LR: {current_lr:.6f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(save_dir, "best_model.pt")
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'tokens_seen': tokens_seen,
                    }, checkpoint_path)
                    print(f"  ✓ En iyi model kaydedildi! Val Loss: {val_loss:.4f}")
                
                # Generate sample
                if global_step % (eval_freq * 5) == 0:
                    print(f"\n  📝 Örnek metin üretimi:")
                    generated = generate_text(model, tokenizer, device, start_context, max_new_tokens=50)
                    print(f"  '{generated}'\n")
                    model.train()
        
        # Epoch sonu
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs} Tamamlandı | Ortalama Loss: {avg_epoch_loss:.4f}")
        print(f"{'='*60}\n")
        
        # Epoch checkpoint
        checkpoint_path = os.path.join(save_dir, f"epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses[-1] if train_losses else 0,
            'val_loss': val_losses[-1] if val_losses else 0,
            'tokens_seen': tokens_seen,
        }, checkpoint_path)
    
    return train_losses, val_losses, track_tokens_seen


def plot_losses(train_losses, val_losses, tokens_seen, save_path):
    """Loss grafiğini çiz"""
    plt.figure(figsize=(10, 6))
    plt.plot(tokens_seen, train_losses, label='Training Loss', linewidth=2)
    plt.plot(tokens_seen, val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Tokens Seen')
    plt.ylabel('Loss')
    plt.title('Luna-LM Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ Loss grafiği kaydedildi: {save_path}")


# ==================== MAIN ====================

def main():
    print("\n" + "="*60)
    print("LUNA-LM FOUNDATION MODEL EĞİTİMİ")
    print("="*60 + "\n")
    
    # 1. Hyperparameters
    print("1. Hyperparameter konfigürasyonu...")
    
    MODEL_SIZE = "small"  # "tiny", "mini", "small", "medium"
    
    config = MODEL_CONFIGS[MODEL_SIZE]

    
    # Training hyperparameters (3.3GB corpus için optimize edildi)
    BATCH_SIZE = 4
    CONTEXT_LENGTH = 512
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-4
    EVAL_FREQ = 100
    EVAL_ITER = 10
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print(f"  Model size: {MODEL_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Context length: {CONTEXT_LENGTH}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    
    # 2. Tokenizer yükle
    print("\n2. Tokenizer yükleniyor...")
    tokenizer = PretrainedTurkishTokenizer('dbmdz/bert-base-turkish-cased')
    vocab_size = tokenizer.vocab_size
    print(f"  ✓ Vocab size: {vocab_size:,}")
    
    # 3. Data yükle (Memory-efficient)
    print("\n3. Corpus yükleniyor (memory-efficient mode)...")
    
    # Proje kök dizini
    project_root = os.path.join(os.path.dirname(__file__), '..')
    corpus_path = os.path.join(project_root, 'foundation_corpus_clean.txt')
    
    MAX_LINES = 500000
    
    lines = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= MAX_LINES:
                break
            line = line.strip()
            if line:
                lines.append(line)
    
    print(f"  ✓ Yüklenen satır: {len(lines):,} (limit: {MAX_LINES:,})")
    
    # Train/Val split (90/10) — RANDOM SPLIT
    import random
    random.seed(42)
    random.shuffle(lines)
    
    split_idx = int(0.9 * len(lines))
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]
    
    train_text = '\n'.join(train_lines)
    val_text = '\n'.join(val_lines)
    
    del lines, train_lines, val_lines
    
    print(f"  ✓ Train: {len(train_text):,} karakter")
    print(f"  ✓ Val: {len(val_text):,} karakter")
    
    # 4. DataLoader oluştur
    print("\n4. DataLoader oluşturuluyor...")
    train_loader = create_dataloader_pretrained(
        train_text, tokenizer, 
        batch_size=BATCH_SIZE, 
        max_length=CONTEXT_LENGTH,
        stride=CONTEXT_LENGTH,
        shuffle=True
    )
    
    val_loader = create_dataloader_pretrained(
        val_text, tokenizer,
        batch_size=BATCH_SIZE,
        max_length=CONTEXT_LENGTH,
        stride=CONTEXT_LENGTH,
        shuffle=False
    )
    
    print(f"  ✓ Train batches: {len(train_loader):,}")
    print(f"  ✓ Val batches: {len(val_loader):,}")
    
    # 5. Model oluştur
    print("\n5. Model oluşturuluyor...")
    
    model_config = {
        "vocab_size": vocab_size,
        "context_length": CONTEXT_LENGTH,
        "emb_dim": config["emb_dim"],
        "n_heads": config["n_heads"],
        "n_layers": config["n_layers"],
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    
    model = GPTModel(model_config)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  ✓ Model hazır!")
    print(f"    Toplam parametreler: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"    Eğitilebilir parametreler: {trainable_params:,}")
    print(f"    Layers: {model_config['n_layers']}")
    print(f"    Heads: {model_config['n_heads']}")
    print(f"    Embedding dim: {model_config['emb_dim']}")
    
    # 6. Optimizer & Scheduler
    print("\n6. Optimizer ayarlanıyor...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    num_training_steps = len(train_loader) * NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_training_steps,
        eta_min=LEARNING_RATE * 0.1
    )
    print(f"  ✓ AdamW optimizer (lr={LEARNING_RATE})")
    print(f"  ✓ Cosine scheduler ({num_training_steps:,} steps)")
    
    # 7. Save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(project_root, f"checkpoints/pretrain_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n7. Checkpoint dizini: {save_dir}")
    
    # Config'i kaydet
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_config': model_config,
            'training_config': {
                'batch_size': BATCH_SIZE,
                'num_epochs': NUM_EPOCHS,
                'learning_rate': LEARNING_RATE,
                'model_size': MODEL_SIZE,
            },
            'tokenizer': 'dbmdz/bert-base-turkish-cased',
            'timestamp': timestamp,
        }, f, indent=2)
    print(f"  ✓ Config kaydedildi: {config_path}")
    
    # 8. TRAINING!
    print("\n" + "="*60)
    print("8. EĞİTİM BAŞLIYOR!")
    print("="*60)
    
    train_losses, val_losses, tokens_seen = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        eval_freq=EVAL_FREQ,
        eval_iter=EVAL_ITER,
        save_dir=save_dir,
        tokenizer=tokenizer,
        start_context="Bugün hava",
        scheduler=scheduler
    )
    
    # 9. Loss grafiği
    print("\n9. Sonuçlar kaydediliyor...")
    plot_path = os.path.join(save_dir, "training_loss.png")
    plot_losses(train_losses, val_losses, tokens_seen, plot_path)
    
    # 10. Final test
    print("\n10. Final test — Metin üretimi:")
    test_prompts = [
        "Bugün hava",
        "Yapay zekâ",
        "Tarih boyunca",
        "İnsan beyni"
    ]
    
    for prompt in test_prompts:
        generated = generate_text(model, tokenizer, device, prompt, max_new_tokens=50)
        print(f"\n  Prompt: '{prompt}'")
        print(f"  Generated: '{generated}'")
    
    # Özet
    print("\n" + "="*60)
    print("EĞİTİM TAMAMLANDI! 🎉")
    print("="*60)
    print(f"\n✓ Checkpoints: {save_dir}/")
    print(f"✓ En iyi model: {save_dir}/best_model.pt")
    print(f"✓ Config: {save_dir}/config.json")
    print(f"✓ Loss grafiği: {save_dir}/training_loss.png")
    print(f"\nToplam tokens işlendi: {tokens_seen[-1]:,}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss: {val_losses[-1]:.4f}")


if __name__ == "__main__":
    main()
