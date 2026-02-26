# Luna-LM SFT (Supervised Fine-Tuning)

Bu klasör, pretrained Luna-LM modelini soru-cevap formatına uyarlamak için SFT pipeline'ını içerir.

## Kullanım

### 1. SFT Verisini Oluştur

```bash
python sft/generate_sft_data.py
```

Bu komut `sft/sft_dataset.jsonl` dosyasını oluşturur. Her satır:
```json
{"system": "...", "user": "Soru", "assistant": "Cevap"}
```

### 2. SFT Eğitimi

```bash
python sft/train_sft.py
```

Pretrained model üzerine SFT eğitimi yapar. Checkpoint'ler `checkpoints/sft_<timestamp>/` altına kaydedilir.

### 3. Test

```bash
python scripts/test_model.py
```

## Prompt Formatı

```
<system>Senin adın Luna. ...</system>
<user>Kullanıcı sorusu</user>
<assistant>Model cevabı</assistant>
```

## Hiperparametreler

| Parametre | Değer | Açıklama |
|---|---|---|
| Learning Rate | 5e-5 | Küçük LR — pretrained ağırlıkları korur |
| Epochs | 10 | Küçük dataset = daha fazla epoch |
| Batch Size | 4 | VRAM'e göre ayarlanabilir |
| Max Length | 512 | Pretrained context length ile aynı |
