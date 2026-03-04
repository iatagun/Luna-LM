"""
Luna-LM DPO Annotation Tool
============================
Human-in-the-loop tercih veri seti oluşturma aracı.

Kullanım:
    python annotate.py                        # Varsayılan checkpoint ile
    python annotate.py --checkpoint <yol>     # Belirli checkpoint
    python annotate.py --task summarization   # Sadece özetleme promptları
    python annotate.py --stats                # İstatistikleri göster

Çıktı:
    dpo_dataset/preferences.jsonl  — DPO eğitim verisi
    dpo_dataset/skipped.jsonl      — Atlanan promptlar
    dpo_dataset/stats.json         — Oturum istatistikleri
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import json
import glob
import random
import argparse
import datetime
from pathlib import Path

import torch
from luna.utils import load_model
from luna.generate import generate_text


# ==========================================
# AYARLAR
# ==========================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "dpo_dataset")
PREFS_FILE = os.path.join(OUTPUT_DIR, "preferences.jsonl")
SKIP_FILE  = os.path.join(OUTPUT_DIR, "skipped.jsonl")
STATS_FILE = os.path.join(OUTPUT_DIR, "stats.json")

# Dual temperature: güvenli vs yaratıcı
TEMP_SAFE       = 0.2   # Düzgün ama sıkıcı
TEMP_CREATIVE   = 0.9   # Yaratıcı ama saçmalayabilir

TASK_TYPES = ["qa", "summarization", "rewrite", "reasoning", "story", "other"]

REJECT_REASONS = [
    "hallucination",   # Hayal gördü / uydurdu
    "too_long",        # Gereksiz uzun
    "too_short",       # Çok kısa, yetersiz
    "off_topic",       # Konudan saptı
    "wrong",           # Yanlış bilgi
    "generic",         # Çok genel, özgün değil
    "repetitive",      # Tekrar yapıyor
    "incoherent",      # Anlamsız
    "good",            # İkisi de iyiydi (selected > rejected için)
]

# Görev türlerine göre örnek promptlar
SAMPLE_PROMPTS = {
    "qa": [
        "Türkiye'nin başkenti neresidir?",
        "Fotosentez nedir?",
        "DNA nedir?",
        "Yapay zeka nasıl çalışır?",
        "Osmanlı İmparatorluğu ne zaman kuruldu?",
        "Dünya'nın en yüksek dağı hangisidir?",
        "Ampulü kim icat etti?",
        "Güneş sistemi kaç gezegenden oluşur?",
        "Türk Kurtuluş Savaşı ne zaman başladı?",
        "İnsanlık tarihinin en eski medeniyeti hangisidir?",
    ],
    "summarization": [
        "Lütfen şu metni özetleyin: Yapay zeka (YZ), makinelerin insan zekasını taklit etmesini sağlayan teknolojileri kapsar. Makine öğrenmesi, derin öğrenme ve doğal dil işleme YZ'nin alt dallarıdır. YZ, sağlık, finans, ulaşım ve eğitim gibi pek çok sektörde devrim yaratmaktadır.",
        "Şu paragrafı özet olarak yeniden yaz: İklim değişikliği, dünya genelinde sıcaklıkların artmasına, deniz seviyelerinin yükselmesine ve aşırı hava olaylarının sıklaşmasına yol açmaktadır. Bunun başlıca nedeni fosil yakıt kullanımından kaynaklanan CO2 emisyonlarıdır. Bilim insanları, küresel sıcaklık artışını 1,5°C'nin altında tutmak için acil önlemler alınması gerektiğini vurgulamaktadır.",
    ],
    "rewrite": [
        "Bu cümleyi profesyonel bir e-posta olarak yeniden yaz: 'Bu işi yapmak istemiyorum.'",
        "Şu cümleyi daha resmi bir dille ifade et: 'Toplantıya katılamayacağım.'",
        "Bu metni çocuklara uygun şekilde sadeleştir: 'Kuantum mekaniği, atom altı parçacıkların davranışını inceleyen fizik dalıdır.'",
    ],
    "reasoning": [
        "Eğer bir trenin hızı 120 km/s ve 2,5 saat yol gidecekse, kaç km yol gider?",
        "Paul işinden yılda 12.000 dolar kazanıyor. Geliri %12,5 artarsa yeni maaşı ne olur?",
        "Bir sınıfta 30 öğrencinin %60'ı sınavı geçtiyse, kaç öğrenci geçemedi?",
    ],
    "story": [
        "Bir robot ve yaşlı bir adam arasında geçen kısa bir hikaye yaz.",
        "Türkiye'nin geleceğini anlatan 3 cümlelik bir ütopya yaz.",
        "Bir kahramanın yolculuğunu anlatan kısa bir paragraf yaz.",
    ],
}


# ==========================================
# YARDIMCI FONKSİYONLAR
# ==========================================

def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "total_annotated": 0,
        "total_skipped": 0,
        "by_task": {},
        "by_reason": {},
        "sessions": [],
    }

def save_stats(stats):
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

def print_stats(stats):
    print("\n" + "="*60)
    print("📊 ANNOTATION İSTATİSTİKLERİ")
    print("="*60)
    print(f"  ✅ Toplam etiketlenen : {stats['total_annotated']}")
    print(f"  ⏭️  Atlanan            : {stats['total_skipped']}")
    
    if stats["by_task"]:
        print("\n  📁 Görev Türüne Göre:")
        for task, count in sorted(stats["by_task"].items(), key=lambda x: -x[1]):
            print(f"     {task:<20} {count}")
    
    if stats["by_reason"]:
        print("\n  ❌ Ret Nedenine Göre:")
        for reason, count in sorted(stats["by_reason"].items(), key=lambda x: -x[1]):
            print(f"     {reason:<20} {count}")
    print("="*60)

def clean_output(generated, tokenizer):
    """Modelin çıktısından sadece asistan cevabını al"""
    out = generated
    # Tokenizer boşluk artifactlerini temizle
    for tag in ["assistant", "user", "system"]:
        out = out.replace(f"< {tag} >", f"<{tag}>")
        out = out.replace(f"< / {tag} >", f"</{tag}>")
    
    if "<assistant>" in out:
        answer = out.split("<assistant>")[-1]
        for stop in ["</assistant>", "<user>", "<system>", "[SEP]", "</s>"]:
            if stop in answer:
                answer = answer.split(stop)[0]
        return answer.strip()
    return out[-300:].strip()

def generate_pair(model, tokenizer, device, prompt, system_prompt):
    """İki farklı temperature ile cevap çifti oluştur"""
    full_prompt = f"<system>{system_prompt}</system>\n<user>{prompt}</user>\n<assistant>"
    
    results = {}
    for label, temp in [("safe", TEMP_SAFE), ("creative", TEMP_CREATIVE)]:
        with torch.no_grad():
            out = generate_text(
                model, tokenizer, device, full_prompt,
                max_new_tokens=200,
                temperature=temp,
                top_k=40,
                repetition_penalty=1.2,
            )
        results[label] = clean_output(out, tokenizer)
    return results["safe"], results["creative"]

def get_score(label):
    """1-5 arası skor al"""
    while True:
        raw = input(f"  Skor {label} (1-5, Enter=atla): ").strip()
        if raw == "":
            return None
        try:
            s = int(raw)
            if 1 <= s <= 5:
                return s
            print("  ⚠️  1 ile 5 arasında bir sayı girin.")
        except ValueError:
            print("  ⚠️  Geçersiz giriş.")

def get_choice():
    """a/b/s/q seçimi al"""
    while True:
        raw = input("\n  Hangisi daha iyi? (a/b/eşit/s=atla/q=çık): ").strip().lower()
        if raw in ("a", "b", "eşit", "esit", "=", "s", "q"):
            return raw
        print("  ⚠️  a, b, eşit, s veya q girin.")

def get_reason():
    """Ret nedenini al"""
    print("\n  Ret nedeni:")
    for i, r in enumerate(REJECT_REASONS, 1):
        print(f"    {i}. {r}")
    while True:
        raw = input("  Neden (numara veya enter=atla): ").strip()
        if raw == "":
            return None
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(REJECT_REASONS):
                return REJECT_REASONS[idx]
        except ValueError:
            # Doğrudan yazılmış olabilir
            if raw in REJECT_REASONS:
                return raw
        print("  ⚠️  Geçersiz seçim.")

def get_task():
    """Görev türünü seç"""
    print("\n  Görev türü:")
    for i, t in enumerate(TASK_TYPES, 1):
        print(f"    {i}. {t}")
    while True:
        raw = input("  Tür (numara/enter=qa): ").strip()
        if raw == "":
            return "qa"
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(TASK_TYPES):
                return TASK_TYPES[idx]
        except ValueError:
            if raw in TASK_TYPES:
                return raw
        print("  ⚠️  Geçersiz seçim.")


# ==========================================
# ANA ANNOTASYON DÖNGÜSÜ
# ==========================================

def annotate(model, tokenizer, device, task_filter=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stats = load_stats()
    
    SYSTEM = ("Senin adın Luna. Amacın insanlara yardımcı olmak ve sorulara açık, "
              "anlaşılır cevaplar vermektir. Bilgin dahilinde olmayan konularda "
              "dürüstçe bilmediğini belirtirir, uydurma bilgi eklemezsin.")
    
    session_start = datetime.datetime.now().isoformat()
    session_count = 0
    
    print("\n" + "="*60)
    print("🌙 LUNA DPO ANNOTATION TOOL")
    print("="*60)
    print(f"  Çıktı: {OUTPUT_DIR}")
    print(f"  Toplam etiketlenen: {stats['total_annotated']}")
    print("\n  ℹ️  Komutlar:")
    print("    a/b     → A veya B'yi seç")
    print("    eşit    → İkisi de eşit kalitede")
    print("    s       → Atla (zor prompt)")
    print("    q       → Çık ve kaydet")
    print("="*60)
    
    # ==========================================
    # Prompt havuzu oluştur
    # ==========================================
    # 1. more_annotate.json dosyasından yükle (öncelik)
    more_file = os.path.join(OUTPUT_DIR, "more_annotate.json")
    # Aynı zamanda proje kökünde de ara
    if not os.path.exists(more_file):
        more_file = os.path.join(os.path.dirname(__file__), "dpo_dataset", "more_annotate.json")
    
    external_prompts = []
    if os.path.exists(more_file):
        with open(more_file, 'r', encoding='utf-8') as f:
            more_data = json.load(f)
        
        # Zaten annotate edilmiş promptları bul (preferences.jsonl'den)
        done_prompts = set()
        if os.path.exists(PREFS_FILE):
            with open(PREFS_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        done_prompts.add(entry.get('prompt', '').strip())
                    except Exception:
                        pass
        
        for item in more_data:
            p = item.get('prompt', '').strip()
            # chosen/rejected zaten var → DPO-ready, atla
            if 'chosen' in item and 'rejected' in item:
                continue
            # Görev filtresi
            if task_filter and item.get('task') != task_filter:
                continue
            # Zaten tamamlanmış → atla
            if p in done_prompts:
                continue
            external_prompts.append({
                'prompt': p,
                'task': item.get('task', 'other'),
                'hint': item.get('target') or item.get('explanation'),
            })
        
        random.shuffle(external_prompts)
        print(f"  📂 more_annotate.json: {len(external_prompts)} annotation bekleyen prompt yüklendi")
        print(f"  ✅ Daha önce tamamlanmış: {len(done_prompts)} prompt")
    
    # 2. Yerleşik örnek promptlar (fallback)
    builtin_prompts = []
    if task_filter and task_filter in SAMPLE_PROMPTS:
        builtin_prompts = [(p, task_filter) for p in SAMPLE_PROMPTS[task_filter]]
    elif not external_prompts:  # Sadece dış dosya boşsa built-in kullan
        for task, ps in SAMPLE_PROMPTS.items():
            builtin_prompts.extend([(p, task) for p in ps])
        random.shuffle(builtin_prompts)
    
    ext_idx = 0
    bi_idx = 0
    
    while True:
        print(f"\n{'─'*60}")
        
        # Sırayla: önce external, sonra built-in, sonra kullanıcı girişi
        current_hint = None
        if ext_idx < len(external_prompts):
            item = external_prompts[ext_idx]
            default_prompt = item['prompt']
            default_task   = item['task']
            current_hint   = item.get('hint')
            ext_idx += 1
            remaining = len(external_prompts) - ext_idx
            print(f"\n📝 [{remaining} kaldı] Hazır prompt (Enter=kullan, yeni yaz veya q=çık):")
            print(f"   Görev : [{default_task}]")
            # Çok satırlı veya uzun promptlar için özel gösterim
            if '\n' in default_prompt or len(default_prompt) > 120:
                print(f"   Prompt:")
                print("   " + "─"*50)
                for line in default_prompt.split('\n'):
                    print(f"   {line}")
                print("   " + "─"*50)
            else:
                print(f"   Prompt: {default_prompt}")
            if current_hint:
                print(f"   💡 İpucu (doğru cevap): {current_hint}")
            user_input = input("\n❓ Prompt (Enter=yukarıdaki, yeni yaz, q=çık): ").strip()
            if user_input == "":
                prompt = default_prompt
                task = default_task
            elif user_input.lower() == "q":
                break
            else:
                prompt = user_input
                task = None
                current_hint = None
        elif bi_idx < len(builtin_prompts):
            default_prompt, default_task = builtin_prompts[bi_idx]
            bi_idx += 1
            print(f"\n📝 Built-in prompt (Enter=kullan, yeni yaz):")
            print(f"   Görev : [{default_task}]")
            print(f"   Prompt: {default_prompt}")
            user_input = input("\n❓ Prompt (Enter=yukarıdaki, yeni yaz, q=çık): ").strip()
            if user_input == "":
                prompt = default_prompt
                task = default_task
            elif user_input.lower() == "q":
                break
            else:
                prompt = user_input
                task = None
        else:
            user_input = input("\n❓ Yeni prompt (q=çık): ").strip()
            if user_input.lower() == "q":
                break
            if not user_input:
                continue
            prompt = user_input
            task = None
        
        # Görev türü belirle
        if task is None:
            task = get_task()
        
        print(f"\n⚙️  Cevaplar üretiliyor (0.2 vs 0.9)...")
        
        try:
            resp_a, resp_b = generate_pair(model, tokenizer, device, prompt, SYSTEM)
        except Exception as e:
            print(f"  ❌ Üretim hatası: {e}")
            continue
        
        # Karıştır — hangisi safe hangisi creative bilinmesin
        if random.random() < 0.5:
            a_text, b_text = resp_a, resp_b
            a_temp, b_temp = TEMP_SAFE, TEMP_CREATIVE
        else:
            a_text, b_text = resp_b, resp_a
            a_temp, b_temp = TEMP_CREATIVE, TEMP_SAFE
        
        print(f"\n{'─'*40}")
        print(f"[A]\n{a_text}\n")
        print(f"{'─'*40}")
        print(f"[B]\n{b_text}\n")
        print(f"{'─'*40}")
        
        # Skorlar
        score_a = get_score("A")
        score_b = get_score("B")
        
        # Seçim
        choice = get_choice()
        
        if choice == "q":
            break
        
        if choice == "s":
            # Atla
            skip_entry = {
                "prompt": prompt,
                "task": task,
                "skipped": True,
                "timestamp": datetime.datetime.now().isoformat(),
            }
            with open(SKIP_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(skip_entry, ensure_ascii=False) + '\n')
            stats["total_skipped"] += 1
            print("  ⏭️  Atlandı ve kaydedildi.")
            continue
        
        # Ret nedeni
        reason = get_reason()
        
        # Açıklama
        comment = input("\n  Yorum (isteğe bağlı, Enter=atla): ").strip() or None
        
        # Doğru cevap (opsiyonel — ipucu varsa göster)
        if current_hint:
            print(f"\n  Doğru cevap (Enter=ipucunu kullan, yeni yaz veya - =boş bırak):")
            print(f"  💡 İpucu: {current_hint}")
            raw_gt = input("  ✏️  : ").strip()
            if raw_gt == "-":
                ground_truth = None
            elif raw_gt == "":
                ground_truth = current_hint
            else:
                ground_truth = raw_gt
        else:
            print("\n  Doğru cevap (biliyorsan yaz, Enter=atla):")
            ground_truth = input("  ✏️  : ").strip() or None
        
        # Chosen/Rejected belirle
        if choice == "a":
            chosen, rejected = a_text, b_text
            chosen_score = score_a
            rejected_score = score_b
            chosen_temp = a_temp
            rejected_temp = b_temp
        elif choice == "b":
            chosen, rejected = b_text, a_text
            chosen_score = score_b
            rejected_score = score_a
            chosen_temp = b_temp
            rejected_temp = a_temp
        else:  # eşit
            # Eşit durumda yüksek skorlu olanı chosen yap
            if (score_a or 3) >= (score_b or 3):
                chosen, rejected = a_text, b_text
                chosen_score, rejected_score = score_a, score_b
                chosen_temp, rejected_temp = a_temp, b_temp
            else:
                chosen, rejected = b_text, a_text
                chosen_score, rejected_score = score_b, score_a
                chosen_temp, rejected_temp = b_temp, a_temp
        
        # Kaydet
        # Eğer ground_truth girilmişse chosen olarak kullan (DPO için daha güçlü sinyal)
        if ground_truth:
            chosen = ground_truth
        
        entry = {
            "task": task,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_score": chosen_score,
            "rejected_score": rejected_score,
            "chosen_temp": chosen_temp if not ground_truth else "human",
            "rejected_temp": rejected_temp,
            "choice": choice,
            "reason": reason,
            "comment": comment,
            "ground_truth": ground_truth,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        with open(PREFS_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # İstatistik güncelle
        stats["total_annotated"] += 1
        session_count += 1
        stats["by_task"][task] = stats["by_task"].get(task, 0) + 1
        if reason:
            stats["by_reason"][reason] = stats["by_reason"].get(reason, 0) + 1
        
        save_stats(stats)
        
        print(f"\n  ✅ Kaydedildi! (Toplam: {stats['total_annotated']})")
    
    # Oturum sonu
    stats["sessions"].append({
        "start": session_start,
        "end": datetime.datetime.now().isoformat(),
        "count": session_count,
    })
    save_stats(stats)
    
    print(f"\n{'='*60}")
    print(f"✅ Oturum tamamlandı — {session_count} örnek etiketlendi")
    print_stats(stats)


# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Luna DPO Annotation Tool")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint yolu")
    parser.add_argument("--task", type=str, default=None,
                        choices=TASK_TYPES,
                        help="Sadece belirli görev türü için prompt kullan")
    parser.add_argument("--stats", action="store_true",
                        help="Sadece istatistikleri göster")
    args = parser.parse_args()
    
    # Sadece istatistik göster
    if args.stats:
        stats = load_stats()
        print_stats(stats)
        return
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Checkpoint bul
    project_root = os.path.dirname(__file__)
    checkpoint = args.checkpoint
    
    if not checkpoint:
        candidates = []
        for pattern in [
            os.path.join(project_root, "checkpoints", "sft_*"),
            os.path.join(project_root, "checkpoints", "pretrain_*"),
        ]:
            candidates.extend(glob.glob(pattern))
        if candidates:
            checkpoint = sorted(candidates)[-1]
    
    if not checkpoint or not os.path.exists(checkpoint):
        print("❌ Model bulunamadı! --checkpoint parametresi ile belirtin.")
        return
    
    print(f"📦 Model yükleniyor: {checkpoint}")
    model, tokenizer, _ = load_model(checkpoint, device=device)
    model.eval()
    
    annotate(model, tokenizer, device, task_filter=args.task)


if __name__ == "__main__":
    main()
