"""
Luna-LM Dataset ve DataLoader yardımcıları
"""

import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetPretrained(Dataset):
    """Pretrained tokenizer kullanan GPT Dataset"""
    
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize — HuggingFace tokenizer'ların internal limitini bypass et
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
