# Luna-LM: Türkçe Foundation Language Model

Sıfırdan Türkçe dil modeli eğitimi - PyTorch implementasyonu.

---

## 🚀 Hızlı Başlangıç

```bash
# 1. Gereksinimleri yükle
pip install torch transformers tokenizers matplotlib

# 2. Model eğitimini başlat
python train_luna_lm.py

# 3. Eğitilmiş modeli test et
python inference_luna_lm.py
```

**Bu kadar! ✅** İlk eğitim ~2-4 saat sürer.

---

## 📁 Ana Dosyalar

| Dosya | Açıklama |
|-------|----------|
| **[train_luna_lm.py](train_luna_lm.py)** | ⭐ Model eğitimi (buradan başlayın!) |
| **[inference_luna_lm.py](inference_luna_lm.py)** | Model test ve kullanım |
| **[foundation_corpus.txt](foundation_corpus.txt)** | Türkçe eğitim verisi (25K+ satır) |
| **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** | 📖 Detaylı eğitim rehberi |
| **[TURKISH_TOKENIZER_README.md](TURKISH_TOKENIZER_README.md)** | Tokenizer kullanım rehberi |

---

## 🎯 Model Boyutları

| Boyut | Parametreler | RAM | Süre | Durum |
|-------|--------------|-----|------|-------|
| **tiny** | 10M | 2-4 GB | 30-60 dk | Hızlı test |
| **small** | 50M | 4-8 GB | 2-4 saat | ✅ Önerilen |
| **medium** | 150M | 8-16 GB | 6-12 saat | İyi sonuç |

`train_luna_lm.py` içinde `MODEL_SIZE` değiştirin.

---

## 💡 Eğitim Süreci

### 1. Veriyi Hazırla
- `foundation_corpus.txt` (25,832 satır Türkçe metin)
- Otomatik train/val split (%90/%10)

### 2. Tokenizer
- Pretrained Türkçe BERT tokenizer (32K vocab)
- Alternatif: Custom BPE tokenizer eğitebilirsiniz

### 3. Model Mimarisi
- GPT-benzeri transformer decoder
- Multi-head attention
- Autoregressive language modeling

### 4. Eğitim
```
Epoch 1/10 | Step 100 | Train Loss: 8.24 | Val Loss: 8.12
  ✓ En iyi model kaydedildi!
  
  📝 Örnek: "Bugün hava çok güzel ve insanlar..."

Epoch 5/10 | Step 2800 | Train Loss: 4.15 | Val Loss: 4.28
...
```

### 5. Çıktılar
```
luna_lm_checkpoints_20251214_153045/
├── best_model.pt           # En iyi model
├── epoch_1.pt, ...         # Her epoch checkpoint
├── config.json             # Model config
└── training_loss.png       # Loss grafiği
```

---

## 🎮 Model Kullanımı

### Komut Satırı
```bash
python inference_luna_lm.py
```

### Python Kodu
```python
from inference_luna_lm import load_model, generate_text
import torch

# Model yükle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer, config = load_model(
    "luna_lm_checkpoints_XXXXXXXX", 
    device=device
)

# Metin üret
text = generate_text(
    model, tokenizer, device,
    prompt="Yapay zekâ",
    max_new_tokens=100,
    temperature=0.8
)
print(text)
```

### İnteraktif Mod
```
📝 Prompt: Tarih boyunca insanlık
🤖 Luna-LM:
Tarih boyunca insanlık birçok zorlukla karşılaştı. 
Medeniyetler yükseldi, çöktü ve yeniden doğdu...

📝 Prompt: quit
```

---

## 🔧 Özelleştirme

### Model Boyutunu Değiştir
```python
# train_luna_lm.py içinde
MODEL_SIZE = "medium"  # tiny, small, medium
```

### Hyperparameter Ayarla
```python
BATCH_SIZE = 4           # GPU memory'e göre
CONTEXT_LENGTH = 256     # Max sequence length
NUM_EPOCHS = 10          # Eğitim epoch sayısı
LEARNING_RATE = 3e-4     # Öğrenme hızı
```

### Custom Tokenizer Kullan
```python
# Kendi tokenizer'ınızı eğitin
python turkish_tokenizer_huggingface.py

# train_luna_lm.py içinde değiştir
from turkish_tokenizer_huggingface import HFTokenizerWrapper
tokenizer = HFTokenizerWrapper('turkish_hf_tokenizer.json')
```

---

## 📊 Beklenen Sonuçlar

### Loss Değerleri (10 epoch sonrası)
- **Train Loss**: 3.0-3.5
- **Val Loss**: 3.2-3.8

### Metin Kalitesi
- ✅ Türkçe kelimeler ve cümleler üretir
- ✅ Temel gramer kurallarını takip eder
- ✅ Konuya uygun kelime seçer
- ⚠️ Uzun paragraflar için daha fazla eğitim gerekir

---

## 🐛 Sorun Giderme

### GPU Memory Hatası
```python
BATCH_SIZE = 2          # veya 1
MODEL_SIZE = "tiny"     # küçük model
CONTEXT_LENGTH = 128    # kısa context
```

### Yavaş Eğitim
- GPU kullandığınızdan emin olun: `torch.cuda.is_available()`
- CUDA yüklü mü kontrol edin
- Küçük model ile test edin

### Loss Düşmüyor
```python
LEARNING_RATE = 5e-4    # artır
NUM_EPOCHS = 20         # daha fazla epoch
```

### Üretilen Metinler Kötü
- Daha fazla epoch eğitin (loss < 3.0 hedefleyin)
- Temperature ayarlayın (0.7-1.0)
- Daha fazla veri ekleyin

---

## 📚 Detaylı Dokümantasyon

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)**: Tüm eğitim detayları, optimizasyon, sorun giderme
- **[TURKISH_TOKENIZER_README.md](TURKISH_TOKENIZER_README.md)**: Tokenizer seçenekleri ve kullanımı
- **[LLMs-from-scratch/](LLMs-from-scratch/)**: Orijinal kod ve eğitim materyalleri

---

## 🎓 Sonraki Adımlar

### 1. Fine-tuning
Özel görevler için model ince ayarı:
- Text Classification ([ch06](LLMs-from-scratch/ch06/))
- Instruction Following ([ch07](LLMs-from-scratch/ch07/))

### 2. Veri Artırma
Daha fazla Türkçe metin ekleyin:
- Wikipedia
- Haberler
- Kitaplar
- Akademik metinler

### 3. Model Büyütme
```python
MODEL_SIZE = "medium"    # 150M parametre
CONTEXT_LENGTH = 512     # Daha uzun context
NUM_EPOCHS = 20          # Daha fazla epoch
```

### 4. Deployment
- ONNX export
- Quantization (INT8)
- FastAPI ile API servisi
- Streamlit UI

---

## 📈 Örnek Çıktılar

### Epoch 1 (Başlangıç)
```
Prompt: "Bugün hava"
Output: "çok ve en ile bir için..."
```

### Epoch 5 (Gelişme)
```
Prompt: "Bugün hava"
Output: "çok güzel ve insanlar dışarıda yürüyüş yapıyor."
```

### Epoch 10 (İyi Sonuç)
```
Prompt: "Bugün hava"
Output: "çok güzel. Gökyüzü açık ve güneş parlıyor. 
İnsanlar parklarda yürüyüş yapıyor ve çocuklar 
oyun oynuyor."
```

---

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun
3. Commit yapın
4. Push edin
5. Pull Request açın

---

## 📝 Lisans

Bu proje MIT lisansı altındadır. LLMs-from-scratch kodu Apache 2.0 lisansı kullanır.

---

## 🙏 Teşekkürler

- **Sebastian Raschka**: [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) kitabı ve kodu
- **DBMDz**: Türkçe BERT tokenizer
- **Hugging Face**: Transformers ve Tokenizers kütüphaneleri

---

## 📞 İletişim

- **GitHub Issues**: Sorular ve bug raporları
- **Discussions**: Genel tartışmalar ve yardım

---

## ⭐ Hızlı Komutlar

```bash
# Eğitim başlat
python train_luna_lm.py

# Model test et
python inference_luna_lm.py

# Custom tokenizer eğit
python turkish_tokenizer_huggingface.py

# Tokenizer test et
python turkish_tokenizer_pretrained.py

# GPU kontrolü
python -c "import torch; print(torch.cuda.is_available())"
```

---

**Başarılar! 🚀** Luna-LM'inizi eğitin ve kendi Türkçe dil modelinizi oluşturun!
