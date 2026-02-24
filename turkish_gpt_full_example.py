"""
Mevcut GPT kodunu Türkçe tokenizer ile kullanma örneği
LLMs-from-scratch kodlarını adapte eder
"""

import sys
import os

# LLMs-from-scratch modüllerini import et
sys.path.append('LLMs-from-scratch/ch04/01_main-chapter-code')
sys.path.append('LLMs-from-scratch/ch05/01_main-chapter-code')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ==================== TOKENIZER SEÇİMİ ====================
# Aşağıdaki 3 seçenekten birini kullanabilirsiniz:

# SEÇENEK 1: Custom BPE (Kendi tokenizer'ınız)
from turkish_gpt_dataloader import TurkishTokenizer, create_dataloader_turkish

# SEÇENEK 2: Hugging Face Tokenizers (Hızlı & Profesyonel)
# from turkish_tokenizer_huggingface import HFTokenizerWrapper, create_dataloader_hf

# SEÇENEK 3: Pretrained (En hızlı, eğitimsiz)
# from turkish_tokenizer_pretrained import PretrainedTurkishTokenizer, create_dataloader_pretrained


# ==================== GPT MODEL (Orijinal) ====================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# ==================== TÜRKÇE KULLANIM ÖRNEĞİ ====================
def main():
    print("=== Türkçe GPT Modeli - Tam Örnek ===\n")
    
    # 1. Tokenizer'ı yükle
    print("1. Tokenizer hazırlanıyor...")
    
    # SEÇENEK 1: Custom tokenizer (önce eğitilmeli)
    if not os.path.exists('turkish_tokenizer.json'):
        print("   ⚠️  Tokenizer bulunamadı! Önce turkish_tokenizer_training.py çalıştırın")
        print("   Şimdilik örnek vocab_size ile devam ediyoruz...")
        vocab_size = 5000  # Placeholder
    else:
        tokenizer = TurkishTokenizer('turkish_tokenizer.json')
        vocab_size = tokenizer.vocab_size
    
    # SEÇENEK 2 veya 3 kullanıyorsanız:
    # tokenizer = HFTokenizerWrapper('turkish_hf_tokenizer.json')
    # tokenizer = PretrainedTurkishTokenizer('dbmdz/bert-base-turkish-cased')
    # vocab_size = tokenizer.vocab_size
    
    print(f"   ✓ Vocab size: {vocab_size}")
    
    # 2. Model konfigürasyonu
    print("\n2. Model konfigürasyonu oluşturuluyor...")
    GPT_CONFIG_124M = {
        "vocab_size": vocab_size,  # TÜRKÇE VOCAB SIZE!
        "context_length": 256,     # Kısa context (memory için)
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    
    print(f"   Model parametreleri:")
    print(f"     Vocab size: {GPT_CONFIG_124M['vocab_size']}")
    print(f"     Context length: {GPT_CONFIG_124M['context_length']}")
    print(f"     Embedding dim: {GPT_CONFIG_124M['emb_dim']}")
    print(f"     Layers: {GPT_CONFIG_124M['n_layers']}")
    print(f"     Heads: {GPT_CONFIG_124M['n_heads']}")
    
    # 3. Model oluştur
    print("\n3. Model oluşturuluyor...")
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Toplam parametre: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # 4. Corpus yükle
    print("\n4. Corpus yükleniyor...")
    with open('foundation_corpus.txt', 'r', encoding='utf-8') as f:
        corpus = f.read()
    print(f"   ✓ Corpus boyutu: {len(corpus):,} karakter")
    
    # 5. DataLoader oluştur
    if os.path.exists('turkish_tokenizer.json'):
        print("\n5. DataLoader oluşturuluyor...")
        train_loader = create_dataloader_turkish(
            corpus,
            tokenizer,
            batch_size=2,
            max_length=256,
            stride=128
        )
        print(f"   ✓ DataLoader hazır: {len(train_loader)} batch")
        
        # 6. İlk batch ile test
        print("\n6. Model test ediliyor...")
        batch = next(iter(train_loader))
        input_batch, target_batch = batch
        
        print(f"   Input shape: {input_batch.shape}")
        print(f"   Target shape: {target_batch.shape}")
        
        with torch.no_grad():
            logits = model(input_batch)
        
        print(f"   Output logits shape: {logits.shape}")
        print(f"   ✓ Model başarıyla çalıştı!")
        
        # 7. Decode örneği
        print("\n7. Decode örneği:")
        decoded = tokenizer.decode(input_batch[0])
        print(f"   {decoded[:100]}...")
    else:
        print("\n⚠️  Tokenizer yok, DataLoader oluşturulamadı")
        print("   turkish_tokenizer_training.py dosyasını çalıştırarak tokenizer eğitin")
    
    print("\n" + "="*60)
    print("SONRAKI ADIMLAR:")
    print("="*60)
    print("""
    1. Tokenizer eğit:
       python turkish_tokenizer_training.py
       
    2. Bu scripti tekrar çalıştır:
       python turkish_gpt_full_example.py
       
    3. Eğitim başlat:
       - LLMs-from-scratch/ch05/01_main-chapter-code/gpt_train.py
       - Tokenizer parametresini değiştir
       - foundation_corpus.txt ile eğit
    """)


if __name__ == "__main__":
    main()
