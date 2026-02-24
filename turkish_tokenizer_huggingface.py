"""
Hugging Face Tokenizers kullanarak Türkçe tokenizer eğitimi
Çok daha hızlı ve profesyonel bir yöntem
"""

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
import torch
from torch.utils.data import Dataset, DataLoader


def train_turkish_tokenizer_hf(corpus_file, vocab_size=5000, save_path='turkish_hf_tokenizer.json'):
    """Hugging Face Tokenizers ile BPE tokenizer eğit"""
    
    print(f"=== Hugging Face Tokenizers ile Eğitim ===")
    print(f"Vocab size: {vocab_size}\n")
    
    # 1. BPE tokenizer oluştur
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    
    # 2. Pre-tokenizer ayarla (Türkçe karakterleri destekler)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # 3. Trainer oluştur
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|unk|>", "<|pad|>"],
        show_progress=True
    )
    
    # 4. Corpus üzerinde eğit
    print("Eğitim başlıyor...")
    tokenizer.train([corpus_file], trainer)
    
    # 5. Post-processor ekle (decode için)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # 6. Kaydet
    tokenizer.save(save_path)
    print(f"\n✓ Tokenizer kaydedildi: {save_path}")
    
    return tokenizer


class HFTokenizerWrapper:
    """Hugging Face tokenizer'ı GPT kodu ile uyumlu hale getirir"""
    
    def __init__(self, tokenizer_path='turkish_hf_tokenizer.json'):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        print(f"✓ Tokenizer yüklendi: {self.vocab_size} token")
    
    def encode(self, text, allowed_special=None):
        """GPT kodu ile uyumlu encode"""
        encoding = self.tokenizer.encode(text)
        return encoding.ids
    
    def decode(self, token_ids):
        """Token ID'leri metne çevir"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids)


class GPTDatasetHF(Dataset):
    """Hugging Face tokenizer kullanan GPT Dataset"""
    
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize
        token_ids = tokenizer.encode(txt)

        # Sliding window
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_hf(txt, tokenizer, batch_size=4, max_length=256,
                        stride=128, shuffle=True, drop_last=True):
    """Hugging Face tokenizer ile DataLoader"""
    
    dataset = GPTDatasetHF(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last
    )
    return dataloader


# ==================== KULLANIM ÖRNEĞİ ====================
if __name__ == "__main__":
    import os
    
    print("=== Hugging Face Tokenizers Örneği ===\n")
    
    # 1. Tokenizer'ı eğit (ilk kez)
    if not os.path.exists('turkish_hf_tokenizer.json'):
        print("1. Yeni tokenizer eğitiliyor...")
        tokenizer = train_turkish_tokenizer_hf(
            corpus_file='foundation_corpus.txt',
            vocab_size=5000,
            save_path='turkish_hf_tokenizer.json'
        )
    else:
        print("1. Mevcut tokenizer yükleniyor...")
    
    # 2. Tokenizer wrapper oluştur
    print("\n2. Tokenizer wrapper hazırlanıyor...")
    tokenizer = HFTokenizerWrapper('turkish_hf_tokenizer.json')
    
    # 3. Test et
    print("\n3. Test ediliyor...")
    test_texts = [
        "Merhaba dünya! Bugün hava çok güzel.",
        "Yapay zekâ ve derin öğrenme üzerine çalışıyorum.",
        "Türkçe karakterler: ğüşıöçĞÜŞİÖÇ"
    ]
    
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"\nOrijinal: {text}")
        print(f"Encoded ({len(encoded)} token): {encoded[:10]}...")
        print(f"Decoded: {decoded}")
    
    # 4. DataLoader oluştur
    print("\n4. DataLoader test ediliyor...")
    with open('foundation_corpus.txt', 'r', encoding='utf-8') as f:
        corpus = f.read()
    
    dataloader = create_dataloader_hf(
        corpus, 
        tokenizer,
        batch_size=2,
        max_length=256
    )
    
    print(f"✓ DataLoader hazır: {len(dataloader)} batch")
    
    # İlk batch'i göster
    inputs, targets = next(iter(dataloader))
    print(f"\nİlk batch:")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Target shape: {targets.shape}")
    print(f"  Decoded (ilk 100 karakter):")
    print(f"  {tokenizer.decode(inputs[0])[:100]}...")
