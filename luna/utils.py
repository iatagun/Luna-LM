"""
Luna-LM Yardımcı Fonksiyonlar
Model yükleme ve ortak araçlar.
"""

import torch
import json
import os

from luna.model import GPTModel
from luna.tokenizer import PretrainedTurkishTokenizer


def load_model(checkpoint_path, device='cuda'):
    """
    Model ve tokenizer'ı yükle.
    
    checkpoint_path bir klasör (pretraining çıktısı) VEYA .pt dosyası (SFT çıktısı) olabilir.
    
    Returns:
        (model, tokenizer, model_config) tuple
    """
    
    model_config = None
    tokenizer_name = 'dbmdz/bert-base-turkish-cased'  # Varsayılan
    
    # Durum 1: checkpoint_path bir KLASÖR (Pretraining veya SFT çıktısı)
    if os.path.isdir(checkpoint_path):
        config_path = os.path.join(checkpoint_path, "config.json")
        
        # SFT model'i öncelikli, yoksa pretrained
        sft_weights = os.path.join(checkpoint_path, "best_sft_model.pt")
        pretrain_weights = os.path.join(checkpoint_path, "best_model.pt")
        
        if os.path.exists(sft_weights):
            weights_path = sft_weights
        elif os.path.exists(pretrain_weights):
            weights_path = pretrain_weights
        else:
            raise FileNotFoundError(f"Model dosyası bulunamadı: {checkpoint_path}")
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            model_config = config['model_config']
            tokenizer_name = config.get('tokenizer', tokenizer_name)
    
    # Durum 2: checkpoint_path bir DOSYA (.pt — SFT çıktısı)
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

    # Tokenizer yükle
    tokenizer = PretrainedTurkishTokenizer(tokenizer_name)
    model_config['vocab_size'] = tokenizer.vocab_size

    print(f"Model config: {model_config}")
    
    # Model oluştur ve ağırlıkları yükle
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
