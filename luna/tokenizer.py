"""
Mevcut Türkçe modellerden tokenizer kullanma
En hızlı yöntem — eğitim gerektirmez
"""

from transformers import AutoTokenizer
import torch


class PretrainedTurkishTokenizer:
    """Önceden eğitilmiş Türkçe tokenizer wrapper"""
    
    def __init__(self, model_name='dbmdz/bert-base-turkish-cased'):
        """
        Türkçe destekli popüler modeller:
        - 'dbmdz/bert-base-turkish-cased': DBMDz tarafından eğitilmiş
        - 'dbmdz/bert-base-turkish-128k-cased': Daha büyük vocab
        - 'xlm-roberta-base': Çok dilli, Türkçe dahil
        - 'facebook/mbart-large-50': 50 dil destekli
        """
        print(f"Tokenizer yükleniyor: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
        
        # BERT tokenizer'da EOS yok — [SEP] token'ı EOS olarak kullan
        # [SEP] zaten "end of sequence" anlamına gelir (id=102)
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = self.tokenizer.sep_token  # [SEP]
        
        # Pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Kolay erişim için
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        
        print(f"✓ Tokenizer yüklendi: {self.vocab_size} token")
        print(f"  PAD token: {self.tokenizer.pad_token} (id={self.pad_token_id})")
        print(f"  EOS token: {self.tokenizer.eos_token} (id={self.eos_token_id})")
    
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
