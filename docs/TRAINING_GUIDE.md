# Luna-LM Foundation Model Eğitim Rehberi

Türkçe foundation language model (Luna-LM) eğitimi için tam rehber.

---

## 🚀 Hızlı Başlangıç

### 1. Gereksinimleri Yükle

```bash
pip install torch transformers tokenizers matplotlib tqdm
```

### 2. Model Eğitimi Başlat

```bash
python train_luna_lm.py
```

**İlk eğitim ~2-4 saat sürebilir** (GPU'ya bağlı)

### 3. Eğitilmiş Modeli Test Et

```bash
python inference_luna_lm.py
```

---

## 📊 Model Boyutları

Eğitim scripti (`train_luna_lm.py`) içinde `MODEL_SIZE` değişkenini değiştirerek model boyutunu seçebilirsiniz:

| Model Size | Parametreler | RAM Gereksinimi | Eğitim Süresi* | Önerilen Kullanım |
|------------|--------------|-----------------|----------------|-------------------|
| **tiny**   | ~10M         | 2-4 GB          | 30-60 dk       | Hızlı test        |
| **small**  | ~50M         | 4-8 GB          | 2-4 saat       | ✅ **Önerilen**   |
| **medium** | ~150M        | 8-16 GB         | 6-12 saat      | İyi sonuçlar      |

\* GTX 1660 Ti / RTX 2060 seviyesi GPU için yaklaşık süreler

---

## 📁 Dosya Yapısı

```
Luna-LM/
├── foundation_corpus.txt              # Eğitim verisi (25K+ satır)
│
├── train_luna_lm.py                   # ⭐ ANA EĞİTİM SCRİPTİ
├── inference_luna_lm.py               # Model test ve kullanım
│
├── turkish_tokenizer_pretrained.py    # Tokenizer (pretrained)
├── turkish_tokenizer_huggingface.py   # Tokenizer (HF custom)
├── turkish_tokenizer_training.py      # Tokenizer (sıfırdan)
│
└── luna_lm_checkpoints_YYYYMMDD_HHMMSS/  # Eğitim çıktıları
    ├── best_model.pt                  # En iyi model checkpoint
    ├── epoch_1.pt, epoch_2.pt, ...    # Her epoch'un checkpointi
    ├── config.json                     # Model konfigürasyonu
    └── training_loss.png               # Loss grafiği
```

---

## 🎯 Adım Adım Rehber

### Adım 1: Model Boyutu Seç

`train_luna_lm.py` dosyasını açın ve 277. satırı düzenleyin:

```python
MODEL_SIZE = "small"  # "tiny", "small", veya "medium"
```

**Önerilen başlangıç:** `"small"` (50M parametre)

---

### Adım 2: Hyperparameter Ayarları (Opsiyonel)

`train_luna_lm.py` içinde 297-302. satırlarda:

```python
BATCH_SIZE = 4           # GPU memory'e göre ayarlayın
CONTEXT_LENGTH = 256     # Daha uzun context = daha fazla memory
NUM_EPOCHS = 10          # Eğitim epoch sayısı
LEARNING_RATE = 3e-4     # Standart GPT learning rate
EVAL_FREQ = 100          # Her 100 step'te bir değerlendirme
EVAL_ITER = 10           # Değerlendirme için batch sayısı
```

**Memory problemi varsa:**
- `BATCH_SIZE` düşürün (2 veya 1)
- `CONTEXT_LENGTH` düşürün (128)
- `MODEL_SIZE = "tiny"` seçin

---

### Adım 3: Eğitimi Başlat

```bash
python train_luna_lm.py
```

**Eğitim sırasında görecekleriniz:**

```
==============================================================
LUNA-LM FOUNDATION MODEL EĞİTİMİ
==============================================================

1. Hyperparameter konfigürasyonu...
  Device: cuda
  Model size: small
  Batch size: 4
  Context length: 256
  Epochs: 10
  Learning rate: 0.0003

2. Tokenizer yükleniyor...
  ✓ Vocab size: 32,000

3. Corpus yükleniyor...
  ✓ Corpus boyutu: 2,583,200 karakter
  ✓ Train: 2,324,880 karakter
  ✓ Val: 258,320 karakter

4. DataLoader oluşturuluyor...
  ✓ Train batches: 5,680
  ✓ Val batches: 632

5. Model oluşturuluyor...
  ✓ Model hazır!
    Toplam parametreler: 52,428,800 (52.4M)
    Layers: 6
    Heads: 8
    Embedding dim: 512

==============================================================
EĞİTİM BAŞLIYOR
==============================================================

Epoch 1/10 | Step 100 | Train Loss: 8.2456 | Val Loss: 8.1234
  ✓ En iyi model kaydedildi! Val Loss: 8.1234

  📝 Örnek metin üretimi:
  'Bugün hava çok güzel, güneş parlıyor ve insanlar dışarıda...'

Epoch 1/10 | Step 200 | Train Loss: 7.8923 | Val Loss: 7.7654
...
```

**Eğitim süresi:**
- **tiny** model: ~30-60 dakika
- **small** model: ~2-4 saat
- **medium** model: ~6-12 saat

---

### Adım 4: Eğitimi İzleme

#### A. Terminalde Real-time
- Loss değerleri her 100 step'te yazdırılır
- Örnek metin üretimleri her 500 step'te gösterilir

#### B. Loss Grafiği
Eğitim bitince `training_loss.png` oluşur:
- Train loss (mavi)
- Validation loss (turuncu)
- Loss düşüyorsa ✅ iyi gidiyor
- Val loss artıyorsa ⚠️ overfitting

---

### Adım 5: Modeli Test Et

Eğitim bitince:

```bash
python inference_luna_lm.py
```

**Test çıktısı:**

```
==============================================================
LUNA-LM INFERENCE
==============================================================

Device: cuda

Checkpoint dizini: luna_lm_checkpoints_20251214_153045

Model yükleniyor...
  Config yüklendi:
    Vocab size: 32,000
    Layers: 6
    Embedding dim: 512
  ✓ Model yüklendi: best_model.pt
    Epoch: 9
    Global step: 56,800
    Val loss: 4.2345

==============================================================
TEST ÜRETİMLERİ
==============================================================

📝 Prompt: 'Bugün hava çok güzel'
🤖 Luna-LM:
Bugün hava çok güzel, gökyüzü açık ve güneş parlıyor. 
İnsanlar parklarda yürüyüş yapıyor, çocuklar oyun 
oynuyor...

📝 Prompt: 'Yapay zekâ teknolojisi'
🤖 Luna-LM:
Yapay zekâ teknolojisi son yıllarda büyük gelişmeler 
gösterdi. Makine öğrenimi algoritmaları...

İnteraktif moda geçmek ister misiniz? (y/n): y

==============================================================
LUNA-LM İNTERAKTİF MOD
==============================================================

📝 Prompt: Tarih boyunca insanlık
🤖 Luna-LM:
Tarih boyunca insanlık birçok zorlukla karşılaştı...

📝 Prompt: quit
Görüşmek üzere! 👋
```

---

## 🎛️ İnteraktif Mod Parametreleri

İnteraktif modda `params` yazarak ayarları değiştirebilirsiniz:

```
📝 Prompt: params

Mevcut parametreler:
  max_tokens: 100
  temperature: 0.8
  top_k: 50
  
Yeni max_tokens: 150
Yeni temperature: 1.2
Yeni top_k: 100
✓ Parametreler güncellendi!
```

**Parametre Açıklamaları:**

- **max_tokens**: Üretilecek maksimum kelime sayısı (50-500 arası)
- **temperature**: Yaratıcılık seviyesi
  - `0.1-0.5`: Deterministik, tutarlı
  - `0.7-1.0`: Dengeli ✅ **önerilen**
  - `1.0-2.0`: Yaratıcı, çeşitli
- **top_k**: Kelime havuzu büyüklüğü (30-100 arası)

---

## 📈 Eğitim İyileştirme

### Loss Düşmüyorsa:

1. **Learning rate'i artır:**
   ```python
   LEARNING_RATE = 5e-4  # 3e-4 yerine
   ```

2. **Daha fazla epoch:**
   ```python
   NUM_EPOCHS = 20  # 10 yerine
   ```

3. **Batch size artır** (GPU memory varsa):
   ```python
   BATCH_SIZE = 8  # 4 yerine
   ```

### Overfitting Varsa (Val loss artıyor):

1. **Dropout artır** (`train_luna_lm.py`, model_config):
   ```python
   "drop_rate": 0.2,  # 0.1 yerine
   ```

2. **Weight decay artır:**
   ```python
   optimizer = torch.optim.AdamW(
       model.parameters(), 
       lr=LEARNING_RATE, 
       weight_decay=0.2  # 0.1 yerine
   )
   ```

3. **Daha fazla veri** ekleyin `foundation_corpus.txt`'e

---

## 🔧 Memory Optimizasyonu

### GPU Memory Yetersiz Hatası:

```
RuntimeError: CUDA out of memory
```

**Çözümler:**

1. **Batch size düşür:**
   ```python
   BATCH_SIZE = 2  # veya 1
   ```

2. **Context length düşür:**
   ```python
   CONTEXT_LENGTH = 128  # 256 yerine
   ```

3. **Küçük model seç:**
   ```python
   MODEL_SIZE = "tiny"
   ```

4. **Gradient accumulation** (gelişmiş):
   ```python
   # train_luna_lm.py içinde, optimizer.step() öncesi:
   if (batch_idx + 1) % 4 == 0:  # Her 4 batch'te bir
       optimizer.step()
       optimizer.zero_grad()
   ```

---

## 💾 Checkpoint Kullanımı

### Eğitimi Devam Ettirme

Eğitim yarıda kesildiyse:

```python
# train_luna_lm.py içinde, model oluşturulduktan sonra:

checkpoint = torch.load('luna_lm_checkpoints_XXXXXXXX/epoch_5.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### En İyi Modeli Kullanma

```python
# inference_luna_lm.py otomatik olarak best_model.pt kullanır
# Manuel yükleme için:

model, tokenizer, config = load_model(
    'luna_lm_checkpoints_XXXXXXXX',
    checkpoint_name='best_model.pt'  # veya 'epoch_10.pt'
)
```

---

## 🎓 Sonraki Adımlar

### 1. Fine-tuning (Özel Görevler İçin)

Modelinizi belirli görevler için fine-tune edin:

- **Text Classification**: `LLMs-from-scratch/ch06/`
- **Instruction Following**: `LLMs-from-scratch/ch07/`

### 2. Daha Fazla Veri

`foundation_corpus.txt`'e daha fazla Türkçe metin ekleyin:
- Wikipedia makaleleri
- Kitaplar (telif hakkı olmayan)
- Haberler
- Akademik metinler

### 3. Tokenizer Optimizasyonu

Custom tokenizer ile daha iyi sonuçlar:

```bash
python turkish_tokenizer_huggingface.py  # Custom eğit
```

Sonra `train_luna_lm.py` içinde tokenizer'ı değiştir:

```python
from turkish_tokenizer_huggingface import HFTokenizerWrapper
tokenizer = HFTokenizerWrapper('turkish_hf_tokenizer.json')
```

### 4. Daha Büyük Model

GPU memory yetiyorsa:

```python
MODEL_SIZE = "medium"  # 150M parametre
```

veya custom config:

```python
model_config = {
    "vocab_size": vocab_size,
    "context_length": 512,      # Daha uzun context
    "emb_dim": 1024,            # Daha büyük
    "n_heads": 16,
    "n_layers": 12,             # Daha derin
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

---

## 🐛 Sık Sorunlar

### 1. "No module named 'transformers'"
```bash
pip install transformers
```

### 2. "CUDA out of memory"
- Batch size düşür
- Model size küçült
- Context length azalt

### 3. "Loss NaN oluyor"
- Learning rate düşür: `LEARNING_RATE = 1e-4`
- Gradient clipping kontrol et (zaten var)

### 4. "Eğitim çok yavaş"
- GPU kullandığınızdan emin olun
- `torch.cuda.is_available()` True dönmeli
- Küçük model ile test edin

### 5. "Üretilen metinler anlamsız"
- Daha fazla epoch eğitin
- Loss 3.0'ın altına düşmeli
- Temperature ayarını değiştirin (0.7-1.0)

---

## 📊 Beklenen Sonuçlar

### Loss Değerleri

| Epoch | Train Loss | Val Loss | Metin Kalitesi |
|-------|-----------|----------|----------------|
| 1     | 8.5       | 8.3      | Anlamsız       |
| 3     | 6.2       | 6.1      | Hece/kelime    |
| 5     | 4.8       | 4.9      | Kelime dizileri|
| 10    | 3.2       | 3.5      | Cümleler ✅    |
| 20    | 2.5       | 2.8      | Mantıklı metinler 🎉 |

**Not:** foundation_corpus.txt boyutuna göre değişir

---

## 📚 Ek Kaynaklar

- **LLMs from Scratch Kitabı**: [GitHub](https://github.com/rasbt/LLMs-from-scratch)
- **Transformer Paper**: "Attention is All You Need"
- **GPT-2 Paper**: "Language Models are Unsupervised Multitask Learners"

---

## ✅ Kontrol Listesi

Eğitime başlamadan önce:

- [ ] `foundation_corpus.txt` dosyası var
- [ ] PyTorch kurulu (`torch.cuda.is_available()` kontrol)
- [ ] transformers paketi kurulu
- [ ] En az 4GB GPU memory (veya tiny model kullan)
- [ ] `train_luna_lm.py` dosyası hazır
- [ ] Model boyutu seçildi

Eğitim sonrası:

- [ ] `best_model.pt` oluştu
- [ ] Loss grafiği düşüş gösteriyor
- [ ] `inference_luna_lm.py` çalışıyor
- [ ] Üretilen metinler mantıklı

---

## 🎉 Başarılar!

Luna-LM'inizi eğitip kullanmaya başladıktan sonra:

1. Fine-tuning ile özel görevlere adapte edin
2. Daha fazla veri ile iyileştirin
3. Hyperparameter tuning yapın
4. Toplulukla paylaşın!

**Sorularınız için:** GitHub Issues veya Discussions kullanabilirsiniz.
