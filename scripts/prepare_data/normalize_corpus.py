"""
Corpus Normalizasyon
Türkçe metin normalizasyonu ve nadir token temizliği.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import re
from collections import Counter


def turkish_lower(text):
    replacement = {
        'I': 'ı',
        'İ': 'i',
        'Ş': 'ş',
        'Ğ': 'ğ',
        'Ü': 'ü',
        'Ö': 'ö',
        'Ç': 'ç'
    }
    for k, v in replacement.items():
        text = text.replace(k, v)
    return text.lower()


def tokenize(text):
    return re.findall(r'\w+|[^\w\s]+', text)


def normalize_corpus(input_file, output_file, min_freq=3):
    print(f"Normalizing {input_file} -> {output_file}")
    print(f"Minimum Frequency: {min_freq} (Tokens < {min_freq} -> <UNK>)")
    
    # Pass 1: Count Frequencies
    print("Pass 1: Counting token frequencies...")
    token_counts = Counter()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_lower = turkish_lower(line)
            tokens = tokenize(line_lower)
            token_counts.update(tokens)
            
    vocab_size = len(token_counts)
    print(f"Total Unique Tokens (Vocab Size): {vocab_size}")
    
    rare_tokens = {token for token, count in token_counts.items() if count < min_freq}
    print(f"Tokens to be replaced with <UNK>: {len(rare_tokens)}")
    print(f"Final Expected Vocab Size: {vocab_size - len(rare_tokens) + 1}")
    
    # Pass 2: Replace and Write
    print("Pass 2: Replacing rare tokens and writing...")
    unk_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            line_lower = turkish_lower(line)
            tokens = tokenize(line_lower)
            
            new_tokens = []
            for token in tokens:
                if token in rare_tokens:
                    new_tokens.append("<UNK>")
                    unk_count += 1
                else:
                    new_tokens.append(token)
            
            fout.write(" ".join(new_tokens) + "\n")
            
    print(f"Total <UNK> insertions: {unk_count}")
    print("Done.")


if __name__ == "__main__":
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    normalize_corpus(
        os.path.join(project_root, "foundation_corpus_clean.txt"),
        os.path.join(project_root, "foundation_corpus_final.txt")
    )
