"""
Luna-LM Model Tanımları
GPT-2 mimarisi - Tüm scriptler bu dosyadan import eder.
"""

import torch
import torch.nn as nn


# ==================== MODEL BİLEŞENLERİ ====================

class MultiHeadAttention(nn.Module):
    mask: torch.Tensor  # Type hint for registered buffer
    
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
        mask_bool: torch.Tensor = self.mask.bool()[:num_tokens, :num_tokens]
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


# ==================== MODEL BOYUTLARI ====================

MODEL_CONFIGS = {
    "tiny": {  # ~2-3M parameters - Çok hızlı test için
        "emb_dim": 128,
        "n_heads": 2,
        "n_layers": 3,
    },
    "mini": {  # ~10M parameters - Küçük corpus için ideal
        "emb_dim": 256,
        "n_heads": 4,
        "n_layers": 4,
    },
    "small": {  # ~50M parameters - Önerilen başlangıç
        "emb_dim": 512,
        "n_heads": 8,
        "n_layers": 6,
    },
    "medium": {  # ~150M parameters - Daha iyi sonuçlar
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 8,
    }
}


def get_model_config(size="small", vocab_size=32000, context_length=512, drop_rate=0.1):
    """Verilen boyuta göre tam model config döndürür."""
    base = MODEL_CONFIGS[size].copy()
    base.update({
        "vocab_size": vocab_size,
        "context_length": context_length,
        "drop_rate": drop_rate,
        "qkv_bias": False,
    })
    return base


# ==================== METİN ÜRETİMİ ====================

def generate_text(model, tokenizer, device, start_text, max_new_tokens=100,
                  temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.2):
    """
    Metin üretimi (Temperature + Top-K + Top-P + Repetition Penalty)
    
    Args:
        model: GPTModel instance
        tokenizer: encode/decode metotları olan tokenizer
        device: torch device
        start_text: Başlangıç metni
        max_new_tokens: Üretilecek maksimum token sayısı
        temperature: Sampling sıcaklığı (0=greedy, düşük=deterministik, yüksek=yaratıcı)
        top_k: Top-k sampling (0=disable)
        top_p: Top-p (nucleus) sampling (1.0=disable)
        repetition_penalty: Tekrar cezası (1.0=disable)
    """
    model.eval()
    
    encoded = tokenizer.encode(start_text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_size = model.pos_emb.weight.shape[0]
            idx_cond = encoded_tensor[:, -context_size:]
            
            logits = model(idx_cond)
            logits = logits[:, -1, :]
            
            # 1. Repetition Penalty
            if repetition_penalty != 1.0:
                for i in range(encoded_tensor.size(0)):
                    for token_id in set(encoded_tensor[i].tolist()):
                        if logits[i, token_id] < 0:
                            logits[i, token_id] *= repetition_penalty
                        else:
                            logits[i, token_id] /= repetition_penalty

            # 2. Temperature
            if temperature > 0:
                logits = logits / temperature
            
            # 3. Top-K Filtering
            if top_k > 0:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                min_value = top_values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_value, 
                                    torch.full_like(logits, float('-inf')), 
                                    logits)
            
            # 4. Top-P (Nucleus) Filtering
            if top_p > 0 and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample
            if temperature > 0:
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
            encoded_tensor = torch.cat((encoded_tensor, idx_next), dim=1)
    
    output_ids = encoded_tensor.squeeze(0).tolist()
    decoded = tokenizer.decode(output_ids)
    return decoded
