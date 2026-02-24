"""
Mevcut Türkçe modellerden tokenizer kullanma
En hızlı yöntem - eğitim gerektirmez
"""

from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader


class PretrainedTurkishTokenizer:
    """Önceden eğitilmiş Türkçe tokenizer wrapper"""
    
    def __init__(self, model_name='bert-base-turkish-cased'):
        """
        Türkçe destekli popüler modeller:
        - 'bert-base-turkish-cased': DBMDz tarafından eğitilmiş
        - 'dbmdz/bert-base-turkish-128k-cased': Daha büyük vocab
        - 'xlm-roberta-base': Çok dilli, Türkçe dahil
        - 'facebook/mbart-large-50': 50 dil destekli
        """
        print(f"Tokenizer yükleniyor: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
        
        # Özel tokenlar ekle
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✓ Tokenizer yüklendi: {self.vocab_size} token")
        print(f"  Pad token: {self.tokenizer.pad_token}")
        print(f"  EOS token: {self.tokenizer.eos_token}")
    
    def encode(self, text, allowed_special=None):
        """GPT kodu ile uyumlu encode"""
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def decode(self, token_ids):
        """Token ID'leri metne çevir"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
    
    def __getitem__(self, key):
        """Tokenizer'a dictionary-style erişim için"""
        return self.tokenizer[key]
    
    def __len__(self):
        """Tokenizer vocab boyutu"""
        return len(self.tokenizer)


class GPTDatasetPretrained(Dataset):
    """Pretrained tokenizer kullanan GPT Dataset"""
    
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize - HuggingFace tokenizer'ların internal limitini bypass et
        # Büyük text'leri chunk'lara bölerek encode et
        chunk_size = 100000  # 100K karakter chunk'lar
        token_ids = []
        
        for i in range(0, len(txt), chunk_size):
            chunk = txt[i:i + chunk_size]
            chunk_tokens = tokenizer.tokenizer.encode(chunk, add_special_tokens=False)
            token_ids.extend(chunk_tokens)
        
        print(f"Toplam token: {len(token_ids)}")

        # Sliding window
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        
        print(f"Oluşturulan sample sayısı: {len(self.input_ids)}")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_pretrained(txt, tokenizer, batch_size=4, max_length=256,
                                 stride=128, shuffle=True, drop_last=True):
    """Pretrained tokenizer ile DataLoader"""
    
    dataset = GPTDatasetPretrained(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last
    )
    return dataloader


# ==================== KULLANIM ÖRNEĞİ ====================
if __name__ == "__main__":
    print("=== Pretrained Türkçe Tokenizer Örneği ===\n")
    
    # Farklı tokenizer seçeneklerini test et
    tokenizer_options = [
        'dbmdz/bert-base-turkish-cased',
        # 'xlm-roberta-base',  # Çok dilli
        # 'facebook/mbart-large-50',  # 50 dil
    ]
    
    for model_name in tokenizer_options:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print('='*60)
        
        # 1. Tokenizer yükle
        tokenizer = PretrainedTurkishTokenizer(model_name)
        
        # 2. Test cümleleri
        print("\n📝 Test Örnekleri:")
        test_texts = [
            "Merhaba dünya! Bugün hava çok güzel.",
            "Yapay zekâ ve derin öğrenme çalışmaları.",
            "Türkçe karakterler: ğüşıöçĞÜŞİÖÇ"
        ]
        
        for text in test_texts:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            print(f"\n  Orijinal: {text}")
            print(f"  Token sayısı: {len(encoded)}")
            print(f"  Encoded: {encoded[:10]}{'...' if len(encoded) > 10 else ''}")
            print(f"  Decoded: {decoded}")
        
        # 3. DataLoader test
        print("\n📊 DataLoader Test:")
        with open('foundation_corpus.txt', 'r', encoding='utf-8') as f:
            corpus = f.read()[:10000]  # İlk 10K karakter
        
        dataloader = create_dataloader_pretrained(
            corpus,
            tokenizer,
            batch_size=2,
            max_length=128
        )
        
        print(f"✓ DataLoader hazır: {len(dataloader)} batch")
        
        # İlk batch'i göster
        inputs, targets = next(iter(dataloader))
        print(f"\n  Batch shape:")
        print(f"    Input: {inputs.shape}")
        print(f"    Target: {targets.shape}")
        print(f"  Decoded (ilk 80 karakter):")
        print(f"    {tokenizer.decode(inputs[0])[:80]}...")
        
        break  # İlk tokenizer yeterli


print("\n" + "="*60)
print("ÖNERİLER:")
print("="*60)
print("""
1. HIZLI BAŞLANGIÇ (Önerilen):
   - dbmdz/bert-base-turkish-cased kullanın
   - Eğitim gerektirmez, hemen kullanıma hazır
   - 32K vocab size

2. DAHA BÜYÜK VOCAB:
   - dbmdz/bert-base-turkish-128k-cased (128K vocab)
   - Daha fazla kelime/subword kapsar

3. ÇOK DİLLİ PROJE İÇİN:
   - xlm-roberta-base (100+ dil)
   - Türkçe + diğer dillerde çalışacaksa

4. CUSTOM TOKENIZER (İleri Seviye):
   - turkish_tokenizer_huggingface.py kullanın
   - Foundation corpus'unuza özel optimize edilir
   - En iyi performans için önerilir
""")
