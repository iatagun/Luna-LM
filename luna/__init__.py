"""
Luna-LM — Türkçe GPT-2 Dil Modeli
"""

__version__ = "1.0.0"

# Çekirdek bileşenler (her zaman erişilebilir)
from luna.model import GPTModel, MODEL_CONFIGS, get_model_config
from luna.generate import generate_text

# Aşağıdakiler doğrudan kullanılabilir, ama transformers gerektirir:
#   from luna.tokenizer import PretrainedTurkishTokenizer
#   from luna.data import GPTDatasetPretrained, create_dataloader_pretrained
#   from luna.utils import load_model
