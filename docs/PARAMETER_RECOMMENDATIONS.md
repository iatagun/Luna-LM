# 🎯 Luna-LM Parametre Önerileri

## 📊 Corpus Analizi

### İstatistikler
```
Toplam Satır:       25,832
Toplam Kelime:      506,744
Toplam Token:       839,275 (~840K)
Unique Kelime:      112,246
Karakter/Token:     4.55
Ortalama Satır:     147.7 karakter
Ortalama Kelime:    6.5 karakter
```

### İçerik Dağılımı
- ✅ **Çok çeşitli içerik**: Felsefe, bilim, AI, fizik, tarih, psikoloji, diyaloglar
- ✅ **Doğal Türkçe**: Günlük konuşmalar, blog tarzı, formal bilimsel metin
- ✅ **Zengin kelime dağarcığı**: 112K unique kelime (çok iyi!)
- ✅ **Uzun bağlamlar**: Ortalama 580 token/satır

---

## ⚙️ Optimized Parametreler

### **ÖNCE (Eski Parametreler)**
```python
MODEL_SIZE = "small"        # 50M params
BATCH_SIZE = 4
CONTEXT_LENGTH = 256
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
EVAL_FREQ = 100
EVAL_ITER = 10
stride = CONTEXT_LENGTH // 2  # 128
```

### **SONRA (Yeni Parametreler)** ✅
```python
MODEL_SIZE = "small"        # 50M params (aynı)
BATCH_SIZE = 2              # ↓ GTX 1650 için güvenli
CONTEXT_LENGTH = 512        # ↑ Daha uzun bağlam
NUM_EPOCHS = 15             # ↑ Küçük veri için daha fazla epoch
LEARNING_RATE = 5e-4        # ↑ Daha agresif başlangıç
EVAL_FREQ = 50              # ↓ Daha sık değerlendirme
EVAL_ITER = 20              # ↑ Daha iyi loss tahmini
stride = CONTEXT_LENGTH * 3 // 4  # 384 (daha fazla overlap)
```

### **Yeni Eklenen: Learning Rate Scheduler** 🆕
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=num_training_steps,
    eta_min=LEARNING_RATE * 0.1  # Son LR = 5e-5
)
```

---

## 📈 Değişiklik Gerekçeleri

### 1. **CONTEXT_LENGTH: 256 → 512**
**Neden?**
- Corpus'ta ortalama satır ~580 token
- 256 ile uzun cümleleri kesiyorduk
- 512 ile tam cümleleri öğrenebilir

**Etki:**
- ✅ Daha iyi bağlam anlama
- ✅ Uzun metinleri doğru modelleyebilme
- ⚠️ Daha fazla GPU memory

### 2. **BATCH_SIZE: 4 → 2**
**Neden?**
- GTX 1650 = 4GB VRAM
- Context 512 olunca memory tüketimi artar
- Batch 2 güvenli, training kararlı

**Etki:**
- ✅ Out of memory riski yok
- ⚠️ Training biraz daha yavaş (ama güvenli)

### 3. **NUM_EPOCHS: 10 → 15**
**Neden?**
- 840K token küçük bir corpus
- Daha fazla epoch = daha iyi öğrenme
- Overfitting riski düşük (veri çeşitli)

**Etki:**
- ✅ Model daha iyi öğrenir
- ⚠️ Training süresi %50 artar

### 4. **LEARNING_RATE: 3e-4 → 5e-4 + Scheduler**
**Neden?**
- Daha yüksek başlangıç LR = hızlı öğrenme
- Cosine scheduler = yumuşak düşüş (3e-4 → 5e-5)
- Overfitting'i önler

**Etki:**
- ✅ Daha hızlı convergence
- ✅ Daha stabil training
- ✅ Loss grafiği daha düzgün

### 5. **EVAL_FREQ: 100 → 50**
**Neden?**
- 840K token ile ~600 step/epoch
- Her 50 step = epoch başına 12 evaluation
- Daha iyi training monitoring

**Etki:**
- ✅ Loss değişimlerini erken fark edersin
- ✅ Overfitting'i hemen görebilirsin
- ⚠️ Biraz daha yavaş (ama değer)

### 6. **Stride: 128 → 384 (3/4 overlap)**
**Neden?**
- Daha fazla overlap = daha fazla training sample
- Model aynı metni farklı pozisyonlardan görür
- Data augmentation etkisi

**Etki:**
- ✅ ~2x daha fazla training sample
- ✅ Daha iyi generalization

---

## 🕐 Beklenen Training Süresi

### Hesaplama:
```
Batch size: 2
Context: 512
Tokens per batch: 2 * 512 = 1,024
Total tokens: 840K
Batches per epoch: ~1,200 (stride ile artış)
Total batches: 1,200 * 15 = 18,000
```

### GTX 1650 ile Tahmini Süre:
- **Batch/s**: ~0.5-1 (512 context ile)
- **Epoch süresi**: 20-40 dakika
- **Total training**: **5-10 saat**

### Loss Beklentileri:
| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1     | 6.5-7.0   | 6.8-7.2  |
| 5     | 3.5-4.0   | 3.8-4.2  |
| 10    | 2.8-3.2   | 3.0-3.5  |
| 15    | 2.5-2.8   | 2.8-3.2  |

---

## 🚀 Eğitimi Başlatma

```bash
python train_luna_lm.py
```

### Eğitim Sırasında İzle:
1. **Loss değerleri**: Train < Val olmalı (ama çok fark olmamalı)
2. **Learning Rate**: Her step'te yavaş düşmeli (5e-4 → 5e-5)
3. **Generated text**: Her 250 step'te örnek üretim
4. **GPU memory**: `nvidia-smi` ile kontrol et

---

## 📝 İyileştirme Seçenekleri

### Eğer GPU Memory Yetmezse:
```python
BATCH_SIZE = 1
CONTEXT_LENGTH = 384
```

### Eğer Daha Hızlı İstersen:
```python
MODEL_SIZE = "tiny"  # 10M params
NUM_EPOCHS = 10
```

### Eğer Daha İyi Sonuç İstersen:
```python
MODEL_SIZE = "medium"  # 150M params
NUM_EPOCHS = 20
LEARNING_RATE = 3e-4  # Daha büyük model = daha küçük LR
```

---

## 🎯 Sonraki Adımlar

### Eğitim Bittikten Sonra:
1. **Loss grafiğini incele** (`training_loss.png`)
2. **Test et** (`python inference_luna_lm.py`)
3. **Farklı promptlar dene**:
   ```
   "Yapay zekâ"
   "Bugün hava"
   "Tarih boyunca"
   "İnsan beyni"
   ```
4. **Fine-tuning için** ch06/ch07'ye bak

### Eğer Sonuçlar İyi Değilse:
- **Overfitting**: EVAL_ITER'i artır, dropout=0.2 yap
- **Underfitting**: NUM_EPOCHS'u artır, MODEL_SIZE'ı büyüt
- **Memory hatası**: BATCH_SIZE=1, CONTEXT_LENGTH=384

---

## 🔥 Kritik İpuçları

1. ✅ **İlk 3 epoch kritik**: Loss hızla düşmeli
2. ✅ **Val loss train'den 0.2-0.4 yüksek olmalı**: Normal
3. ⚠️ **Val loss artmaya başlarsa**: Overfitting, dur
4. ✅ **Generated text her epoch daha iyi olmalı**: Kalite göstergesi
5. 🔄 **Checkpoint'leri sakla**: En iyi model = en düşük val loss

---

## 📊 Model Karşılaştırması

| Model Size | Params | Context | Batch | Training Time | Expected Loss |
|-----------|--------|---------|-------|---------------|---------------|
| tiny      | 10M    | 512     | 4     | 2-3 saat      | 3.0-3.5       |
| **small** | **50M**| **512** | **2** | **5-10 saat** | **2.5-3.0**   |
| medium    | 150M   | 512     | 1     | 15-20 saat    | 2.0-2.5       |

**Tavsiye**: `small` ile başla, sonuçlar iyiyse `medium`'a geç!

---

## 🎓 Referanslar

Bu parametreler şu kaynaklara göre optimize edildi:
- **LLMs-from-scratch Ch05**: Training best practices
- **Corpus analysis**: 840K token, 112K vocab
- **GPU constraints**: GTX 1650 4GB VRAM
- **Turkish tokenizer**: dbmdz/bert-base-turkish-cased

---

## 💡 Son Öneriler

1. **Sabırlı ol**: 5-10 saat sürecek
2. **Logları kaydet**: Training çıktısını bir dosyaya yönlendir
   ```bash
   python train_luna_lm.py 2>&1 | tee training.log
   ```
3. **Checkpoint'leri yedekle**: `best_model.pt` çok değerli
4. **Farklı seed'ler dene**: Rastgelelik etkisini gör
5. **Sonuçları paylaş**: Başarılı olursan community'ye katkıda bulun! 🚀

---

**Hazırlandı**: 2025-12-15  
**Corpus**: foundation_corpus.txt (840K tokens)  
**Model**: Luna-LM GPT-small (50M params)  
**Status**: ✅ Ready to train!
