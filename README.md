# Luna-LM: Türkçe Foundation Language Model 🌙

Sıfırdan Türkçe dil modeli eğitimi - GPT mimarisi ile PyTorch implementasyonu.

## 🚀 Hızlı Başlangıç

```bash
# 1. Gereksinimleri yükle
pip install torch transformers tokenizers matplotlib

# 2. Model eğitimini başlat (2-4 saat)
python train_luna_lm.py

# 3. Eğitilmiş modeli kullan
python inference_luna_lm.py
```

## 📖 Dokümantasyon

- **[QUICKSTART.md](QUICKSTART.md)** - Hızlı başlangıç rehberi
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Detaylı eğitim rehberi  
- **[TURKISH_TOKENIZER_README.md](TURKISH_TOKENIZER_README.md)** - Tokenizer kullanımı

## 🎯 Özellikler

- ✅ Türkçe corpus üzerinde pretraining (foundation_corpus.txt)
- ✅ 3 farklı model boyutu (10M, 50M, 150M parametre)
- ✅ Pretrained Türkçe tokenizer (32K vocab)
- ✅ Custom tokenizer eğitimi desteği
- ✅ Real-time training monitoring
- ✅ İnteraktif metin üretimi
- ✅ Checkpoint sistemi
- ✅ GPU & CPU desteği

## 📊 Model Boyutları

| Boyut | Parametreler | Eğitim Süresi | Kullanım |
|-------|--------------|---------------|----------|
| tiny  | ~10M | 30-60 dk | Hızlı test |
| small | ~50M | 2-4 saat | ✅ Önerilen |
| medium | ~150M | 6-12 saat | İyi sonuç |

## 🔧 Proje Yapısı

```
Luna-LM/
├── model.py                            # ⭐ Model mimarisi (GPTModel, generate_text)
├── train_luna_lm.py                    # Foundation model eğitimi
├── test_luna_lm.py                     # Model testi & interaktif sohbet
├── inference_luna_lm.py                # Model inference
├── sft_luna_lm.py                      # Supervised Fine-Tuning
│
├── generate_massive_sft.py             # SFT veri seti üretimi
│
├── turkish_tokenizer_pretrained.py     # Hazır tokenizer wrapper
├── turkish_tokenizer_huggingface.py    # Custom HF tokenizer
├── turkish_tokenizer_training.py       # BPE sıfırdan eğitimi
│
├── requirements.txt                    # Bağımlılıklar
├── QUICKSTART.md                       # Hızlı başlangıç
├── TRAINING_GUIDE.md                   # Detaylı rehber
└── TURKISH_TOKENIZER_README.md         # Tokenizer rehberi
```


## 💡 Kullanım Örneği

```python
from inference_luna_lm import load_model, generate_text
import torch

# Model yükle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer, config = load_model("luna_lm_checkpoints_XXXXXXXX", device=device)

# Metin üret
text = generate_text(
    model, tokenizer, device,
    prompt="Yapay zekâ",
    max_new_tokens=100,
    temperature=0.8
)
print(text)
```

## 📈 Beklenen Sonuçlar

10 epoch sonrası:
- Train Loss: 3.0-3.5
- Val Loss: 3.2-3.8
- Türkçe kelime ve cümleler
- Temel gramer kuralları
- Anlamlı metin üretimi

## 🎓 Sonraki Adımlar

1. **Fine-tuning**: Özel görevler için model ince ayarı
2. **Veri Artırma**: Daha fazla Türkçe metin ekleme
3. **Model Büyütme**: Daha büyük model boyutları
4. **Deployment**: API servisi ve web arayüzü

## 🙏 Teşekkürler

- [Sebastian Raschka](https://sebastianraschka.com/) - LLMs from Scratch
- [DBMDz](https://huggingface.co/dbmdz) - Türkçe BERT tokenizer
- [Hugging Face](https://huggingface.co/) - Transformers kütüphanesi