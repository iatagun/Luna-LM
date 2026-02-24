"""
Türkçe tokenizer ile GPT DataLoader
Mevcut kodları custom tokenizer ile adapte eder
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json


class TurkishTokenizer:
    """SimpleBPETokenizer ile uyumlu wrapper"""
    
    def __init__(self, tokenizer_path='turkish_tokenizer.json'):
        self.load(tokenizer_path)
    
    def load(self, path):
        """Kaydedilmiş tokenizer'ı yükle"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.vocab_size = data['vocab_size']
        self.merges = {tuple(pair): i for i, pair in enumerate(data['merges'])}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Özel tokenlar ekle
        if '<|endoftext|>' not in self.vocab:
            self.vocab['<|endoftext|>'] = len(self.vocab)
            self.inverse_vocab[len(self.inverse_vocab)] = '<|endoftext|>'
        
        print(f"✓ Tokenizer yüklendi: {len(self.vocab)} token")
    
    def encode(self, text, allowed_special=None):
        """GPT kodu ile uyumlu encode fonksiyonu"""
        import re
        
        # Özel tokenleri işle
        if allowed_special and '<|endoftext|>' in text:
            parts = text.split('<|endoftext|>')
            result = []
            for i, part in enumerate(parts):
                if part:
                    result.extend(self._encode_text(part))
                if i < len(parts) - 1:
                    result.append(self.vocab['<|endoftext|>'])
            return result
        else:
            return self._encode_text(text)
    
    def _encode_text(self, text):
        """İç encode fonksiyonu"""
        import re
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        words = [' '.join(list(word) + ['</w>']) for word in words]
        
        tokens = []
        for word in words:
            word_tokens = self._tokenize_word(word)
            tokens.extend([self.vocab.get(t, 0) for t in word_tokens])
        
        return tokens
    
    def _tokenize_word(self, word):
        """Kelimeyi tokenize et"""
        symbols = word.split()
        
        for pair in self.merges:
            bigram = ' '.join(pair)
            replacement = ''.join(pair)
            word = ' '.join(symbols)
            word = word.replace(bigram, replacement)
            symbols = word.split()
            
        return symbols
    
    def decode(self, token_ids):
        """Token ID'leri metne çevir"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        tokens = [self.inverse_vocab.get(tid, '') for tid in token_ids]
        text = ''.join(tokens).replace('</w>', ' ')
        text = text.replace('<|endoftext|>', '\n')
        return text.strip()


class GPTDatasetV1Turkish(Dataset):
    """Türkçe tokenizer kullanan GPT Dataset"""
    
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize işlemi
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Sliding window ile chunk'lara ayır
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_turkish(txt, tokenizer, batch_size=4, max_length=256,
                              stride=128, shuffle=True, drop_last=True, 
                              num_workers=0):
    """Türkçe tokenizer ile DataLoader oluştur"""
    
    # Dataset oluştur
    dataset = GPTDatasetV1Turkish(txt, tokenizer, max_length, stride)

    # DataLoader oluştur
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last, 
        num_workers=num_workers
    )

    return dataloader


# ==================== KULLANIM ÖRNEĞİ ====================
if __name__ == "__main__":
    print("=== Türkçe GPT DataLoader Test ===\n")
    
    # 1. Tokenizer'ı yükle
    print("1. Tokenizer yükleniyor...")
    tokenizer = TurkishTokenizer('turkish_tokenizer.json')
    
    # 2. Corpus'u yükle
    print("\n2. Corpus yükleniyor...")
    with open('foundation_corpus.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Corpus boyutu: {len(text)} karakter")
    
    # 3. DataLoader oluştur
    print("\n3. DataLoader oluşturuluyor...")
    dataloader = create_dataloader_turkish(
        text, 
        tokenizer,
        batch_size=2, 
        max_length=256,
        stride=128
    )
    
    print(f"✓ DataLoader hazır: {len(dataloader)} batch")
    
    # 4. İlk batch'i göster
    print("\n4. İlk batch:")
    dataiter = iter(dataloader)
    inputs, targets = next(dataiter)
    
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"\nİlk input sample (ilk 10 token):")
    print(inputs[0, :10])
    print(f"\nDecode edilmiş (ilk 50 karakter):")
    print(tokenizer.decode(inputs[0])[:50])
