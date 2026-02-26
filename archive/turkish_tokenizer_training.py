"""
Türkçe corpus üzerinde BPE tokenizer eğitimi
foundation_corpus.txt kullanarak custom tokenizer oluşturur
"""

import re
from collections import Counter


class SimpleBPETokenizer:
    """Basitleştirilmiş BPE Tokenizer - Türkçe için özelleştirilmiş"""
    
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        self.inverse_vocab = {}
        
    def train(self, text):
        """Corpus üzerinde tokenizer eğit"""
        print(f"Eğitim başlıyor... Hedef vocab size: {self.vocab_size}")
        
        # 1. Metni kelimelere ayır
        words = self._get_words(text)
        
        # 2. İlk vocab: Her karakter bir token
        # Türkçe karakterleri de ekle: ğ, ü, ş, ı, ö, ç
        self.vocab = self._initialize_vocab(words)
        print(f"Başlangıç vocab size: {len(self.vocab)}")
        
        # 3. BPE algoritması: En sık görülen çiftleri birleştir
        word_counts = Counter(words)
        num_merges = self.vocab_size - len(self.vocab)
        
        for i in range(num_merges):
            pairs = self._get_stats(word_counts)
            if not pairs:
                break
                
            best_pair = max(pairs, key=pairs.get)
            word_counts = self._merge_vocab(best_pair, word_counts)
            self.merges[best_pair] = len(self.vocab)
            self.vocab[''.join(best_pair)] = len(self.vocab)
            
            if (i + 1) % 100 == 0:
                print(f"Merge {i+1}/{num_merges} tamamlandı")
        
        # Inverse vocab oluştur (decode için)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Eğitim tamamlandı! Final vocab size: {len(self.vocab)}")
        
    def _get_words(self, text):
        """Metni kelimelere ayır"""
        # Türkçe karakterleri koruyarak tokenize et
        pattern = r'\w+|[^\w\s]'
        words = re.findall(pattern, text.lower())
        # Her kelimeyi karakter listesine çevir ve sonuna </w> ekle
        return [' '.join(list(word) + ['</w>']) for word in words]
    
    def _initialize_vocab(self, words):
        """İlk vocabulary oluştur - her karakter bir token"""
        vocab = {}
        for word in words:
            for char in word.split():
                if char not in vocab:
                    vocab[char] = len(vocab)
        return vocab
    
    def _get_stats(self, word_counts):
        """En sık görülen çiftleri say"""
        pairs = Counter()
        for word, count in word_counts.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += count
        return pairs
    
    def _merge_vocab(self, pair, word_counts):
        """Seçilen çifti birleştir"""
        new_word_counts = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, count in word_counts.items():
            new_word = word.replace(bigram, replacement)
            new_word_counts[new_word] = count
        return new_word_counts
    
    def encode(self, text):
        """Metni token ID'lere çevir"""
        words = self._get_words(text)
        tokens = []
        
        for word in words:
            # Her kelimeyi mevcut merge'lere göre böl
            word_tokens = self._tokenize_word(word)
            tokens.extend([self.vocab.get(t, 0) for t in word_tokens])
        
        return tokens
    
    def _tokenize_word(self, word):
        """Bir kelimeyi tokenize et"""
        symbols = word.split()
        
        # Tüm merge'leri uygula
        for pair in self.merges:
            bigram = ' '.join(pair)
            replacement = ''.join(pair)
            word = ' '.join(symbols)
            word = word.replace(bigram, replacement)
            symbols = word.split()
            
        return symbols
    
    def decode(self, token_ids):
        """Token ID'leri metne çevir"""
        tokens = [self.inverse_vocab.get(tid, '') for tid in token_ids]
        text = ''.join(tokens).replace('</w>', ' ')
        return text.strip()
    
    def save(self, path):
        """Tokenizer'ı kaydet"""
        import json
        data = {
            'vocab': self.vocab,
            'merges': [(pair[0], pair[1]) for pair in self.merges.keys()],
            'vocab_size': self.vocab_size
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer kaydedildi: {path}")
    
    def load(self, path):
        """Kaydedilmiş tokenizer'ı yükle"""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.vocab_size = data['vocab_size']
        self.merges = {tuple(pair): i for i, pair in enumerate(data['merges'])}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Tokenizer yüklendi: {path}")


# ==================== KULLANIM ÖRNEĞİ ====================
if __name__ == "__main__":
    print("=== Türkçe BPE Tokenizer Eğitimi ===\n")
    
    # 1. Foundation corpus'u yükle
    print("1. Corpus yükleniyor...")
    with open('foundation_corpus.txt', 'r', encoding='utf-8') as f:
        corpus = f.read()
    
    print(f"Corpus boyutu: {len(corpus)} karakter\n")
    
    # 2. Tokenizer'ı eğit
    print("2. Tokenizer eğitimi başlıyor...")
    tokenizer = SimpleBPETokenizer(vocab_size=5000)
    tokenizer.train(corpus)
    
    # 3. Kaydet
    print("\n3. Tokenizer kaydediliyor...")
    tokenizer.save('turkish_tokenizer.json')
    
    # 4. Test et
    print("\n4. Test ediliyor...")
    test_text = "Merhaba, bugün hava çok güzel. Yapay zekâ çalışıyorum."
    print(f"Test metni: {test_text}")
    
    encoded = tokenizer.encode(test_text)
    print(f"Encoded: {encoded}")
    print(f"Token sayısı: {len(encoded)}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    
    # Vocab'tan örnekler göster
    print("\n5. Vocab'tan örnekler:")
    sample_tokens = list(tokenizer.vocab.items())[:20]
    for token, idx in sample_tokens:
        print(f"  '{token}': {idx}")
