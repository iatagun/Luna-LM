# Türkçe Tokenizer Kullanım Rehberi

Bu dosyalar, Luna-LM projesinde Türkçe tokenizer kullanımı için hazırlanmıştır.

## 🚀 Hızlı Başlangıç

### Yöntem 1: Pretrained Tokenizer (ÖNERİLEN - En Hızlı)

```bash
# Gerekli paketleri yükle
pip install transformers tokenizers

# Test et
python turkish_tokenizer_pretrained.py
```

**Artıları:**
- ✅ Hiç eğitim gerektirmez
- ✅ Anında kullanıma hazır
- ✅ 32K+ Türkçe vocab
- ✅ Profesyonel kalite

**Eksileri:**
- ❌ Foundation corpus'unuza özel değil

---

### Yöntem 2: Hugging Face Tokenizers (Profesyonel)

```bash
# Tokenizer eğit (5-10 dakika)
python turkish_tokenizer_huggingface.py

# Test et
python turkish_gpt_full_example.py
```

**Artıları:**
- ✅ Çok hızlı (Rust tabanlı)
- ✅ Corpus'unuza özel
- ✅ Endüstri standardı
- ✅ Kolay kullanım

**Eksileri:**
- ❌ Ek paket gerektirir: `tokenizers`

---

### Yöntem 3: Sıfırdan BPE (Eğitim Amaçlı)

```bash
# Tokenizer eğit (yavaş olabilir)
python turkish_tokenizer_training.py

# DataLoader ile kullan
python turkish_gpt_dataloader.py

# Tam örnek
python turkish_gpt_full_example.py
```

**Artıları:**
- ✅ BPE algoritmasını öğrenirsiniz
- ✅ Tamamen kontrol sizde
- ✅ Corpus'unuza özel

**Eksileri:**
- ❌ Yavaş (saf Python)
- ❌ Karmaşık kod

---

## 📁 Dosya Açıklamaları

| Dosya | Açıklama |
|-------|----------|
| `turkish_tokenizer_pretrained.py` | Hazır Türkçe tokenizer kullanımı |
| `turkish_tokenizer_huggingface.py` | HF Tokenizers ile eğitim |
| `turkish_tokenizer_training.py` | Sıfırdan BPE implementasyonu |
| `turkish_gpt_dataloader.py` | Custom tokenizer + DataLoader |
| `turkish_gpt_full_example.py` | Tam GPT modeli örneği |

---

## 🎯 Adım Adım: İlk Defa Başlayanlar

### 1. Ortamı Hazırla

```bash
# Proje dizinine git
cd Luna-LM

# Virtual environment aktif et (varsa)
# Windows:
venv\Scripts\activate

# Gerekli paketleri yükle
pip install transformers tokenizers torch
```

### 2. Tokenizer Seç ve Test Et

**Hızlı Test (Önerilen):**
```bash
python turkish_tokenizer_pretrained.py
```

**Custom Eğit:**
```bash
python turkish_tokenizer_huggingface.py
```

### 3. GPT ile Kullan

```bash
python turkish_gpt_full_example.py
```

---

## 🔧 Mevcut Kodları Adapte Etme

### Eski Kod (GPT-2):
```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
token_ids = tokenizer.encode(text)
```

### Yeni Kod (Türkçe - Seçenek 1):
```python
from turkish_tokenizer_pretrained import PretrainedTurkishTokenizer

tokenizer = PretrainedTurkishTokenizer('dbmdz/bert-base-turkish-cased')
token_ids = tokenizer.encode(text)
```

### Yeni Kod (Türkçe - Seçenek 2):
```python
from turkish_tokenizer_huggingface import HFTokenizerWrapper

tokenizer = HFTokenizerWrapper('turkish_hf_tokenizer.json')
token_ids = tokenizer.encode(text)
```

### Yeni Kod (Türkçe - Seçenek 3):
```python
from turkish_gpt_dataloader import TurkishTokenizer

tokenizer = TurkishTokenizer('turkish_tokenizer.json')
token_ids = tokenizer.encode(text)
```

---

## 📊 Karşılaştırma

| Özellik | Pretrained | Hugging Face | Sıfırdan BPE |
|---------|-----------|--------------|--------------|
| **Hız** | ⚡⚡⚡ | ⚡⚡⚡ | ⚡ |
| **Eğitim Süresi** | 0 dk | 5-10 dk | 30-60 dk |
| **Corpus'a Özel** | ❌ | ✅ | ✅ |
| **Vocab Size** | 32K | Ayarlanabilir | Ayarlanabilir |
| **Kullanım Kolaylığı** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Eğitim Değeri** | ⭐ | ⭐⭐ | ⭐⭐⭐ |

---

## 🐛 Sık Karşılaşılan Sorunlar

### 1. "ModuleNotFoundError: No module named 'transformers'"
```bash
pip install transformers
```

### 2. "ModuleNotFoundError: No module named 'tokenizers'"
```bash
pip install tokenizers
```

### 3. "FileNotFoundError: turkish_tokenizer.json"
```bash
# Önce tokenizer'ı eğitin:
python turkish_tokenizer_training.py
# veya
python turkish_tokenizer_huggingface.py
```

### 4. Yavaş Eğitim
- `turkish_tokenizer_huggingface.py` kullanın (çok daha hızlı)
- Veya corpus boyutunu azaltın (test için)

---

## 💡 Öneriler

1. **İlk Kez Başlıyorsanız:** `turkish_tokenizer_pretrained.py` ile başlayın
2. **Production İçin:** `turkish_tokenizer_huggingface.py` ile custom tokenizer eğitin
3. **Öğrenme İçin:** `turkish_tokenizer_training.py` ile BPE algoritmasını anlayın

---

## 📚 Ek Kaynaklar

- [BPE Algoritması Açıklaması](LLMs-from-scratch/ch02/05_bpe-from-scratch/)
- [Hugging Face Tokenizers Docs](https://huggingface.co/docs/tokenizers/)
- [Türkçe BERT Model](https://huggingface.co/dbmdz/bert-base-turkish-cased)

---

## ✅ Başarı Kontrolü

Aşağıdaki kodu çalıştırın:

```python
from turkish_tokenizer_pretrained import PretrainedTurkishTokenizer

tokenizer = PretrainedTurkishTokenizer('dbmdz/bert-base-turkish-cased')
text = "Merhaba dünya! Yapay zekâ çalışıyorum."
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)

print(f"Orijinal: {text}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
print("\n✅ Tokenizer başarıyla çalışıyor!")
```

Eğer hata almadıysanız, hazırsınız! 🎉
