import random
import os

KEYWORDS = {
    "bilim": ["bilim", "deney", "araştırma", "teori", "analiz", "teknoloji"],
    "sanat": ["sanat", "resim", "müzik", "sahne", "edebiyat", "şiir", "yazar"],
    "tarih": ["tarih", "savaş", "dönem", "yüzyıl", "antik", "medeniyet"],
    "doğa": ["doğa", "iklim", "canlı", "bitki", "hayvan", "çevre"],
    "yaşam": ["yaşam", "insan", "toplum", "kültür", "eğitim", "sağlık"]
}

def mine_corpus(filepath, num_samples_per_topic=5):
    results = {k: [] for k in KEYWORDS.keys()}
    file_size = os.path.getsize(filepath)
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        # Try 100 random seeks to find good diverse content
        for _ in range(100):
            pos = random.randint(0, file_size - 10000)
            f.seek(pos)
            f.readline() # partial line
            
            # Read next 20 lines
            chunk = [f.readline().strip() for _ in range(20)]
            
            for line in chunk:
                if len(line) < 50 or len(line) > 200: continue
                if not line[0].isupper() or line[-1] not in ['.', '!', '?']: continue
                
                # Check topics
                line_lower = line.lower()
                for topic, words in KEYWORDS.items():
                    if len(results[topic]) < num_samples_per_topic:
                        if any(w in line_lower for w in words):
                            results[topic].append(line)
                            break # Assign to first matching topic
                            
    return results

data = mine_corpus(r"c:\Users\user\OneDrive\Belgeler\GitHub\Luna-LM\foundation_corpus_clean.txt")

print("MINED SENTENCES:")
for topic, sentences in data.items():
    print(f"\n--- {topic.upper()} ---")
    for s in sentences:
        print(f"{s}")
