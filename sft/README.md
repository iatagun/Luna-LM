# Luna-LM SFT (Supervised Fine-Tuning)

Pretrained Luna-LM modelini soru-cevap formatına uyarlamak için SFT pipeline'ı.

> Referans: [LLMs-from-scratch ch07](../LLMs-from-scratch/ch07) — Sebastian Raschka

## Kullanım

### 1. SFT Verisini Hazırla

JSONL formatında, her satır aşağıdaki alanlardan birini içermeli:

```json
{"user": "Soru", "assistant": "Cevap"}
{"instruction": "Talimat", "input": "", "output": "Cevap"}
{"question": "Soru", "answer": "Cevap"}
```

Dosyayı şu konumlardan birine koyun:
- `sft/sft_dataset.jsonl` (önerilen)
- `sft_dataset_luna_text.jsonl`

Küçük bir demo dataset oluşturmak için:
```bash
python sft/generate_sft_data.py
```

### 2. SFT Eğitimi

```bash
python sft/train_sft.py
```

- Pretrained model otomatik bulunur
- Veri **85/10/5** train/val/test split edilir
- Checkpoint'ler `checkpoints/sft_<timestamp>/` altına kaydedilir
- Loss grafiği otomatik kaydedilir

### 3. Test

```bash
python scripts/test_model.py
```

## Hiperparametreler (83K dataset için)

| Parametre | Değer | Açıklama |
|---|---|---|
| Learning Rate | 5e-5 | ch07 referansı |
| Weight Decay | 0.1 | ch07 referansı |
| Epochs | 2 | 83K × 2 = 166K step |
| Batch Size | 8 | VRAM'e göre: 4/8/16 |
| Max Length | 512 | Pretrained context |
| Eval Freq | 200 steps | Büyük dataset → seyrek eval |
| Warmup | %5 | Linear warmup |
| Scheduler | Cosine | Warmup → Cosine decay |

## Teknik Detaylar

- **EOS Token:** Her örneğin sonuna pad_token_id eklenir (ch07 yaklaşımı)
- **Pad Masking:** İlk EOS tutuluyor, geri kalan pad'ler `-100` ile maskelenir
- **LR Schedule:** Linear warmup (%5) → Cosine annealing
- **Veri formatı:** Otomatik alan tespiti (user/assistant, instruction/output, question/answer)
- **Loss Plot:** `checkpoints/sft_*/sft_loss.png` olarak kaydedilir
