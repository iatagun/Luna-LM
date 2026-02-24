import json
import re

INPUT_FILE = r"c:\Users\user\OneDrive\Belgeler\GitHub\Luna-LM\sft_dataset_luna_text.jsonl"
OUTPUT_FILE = r"c:\Users\user\OneDrive\Belgeler\GitHub\Luna-LM\sft_dataset_luna_text.jsonl" # Overwrite

NEW_SYSTEM_PROMPT = "Senin adın Luna. Amacın insanlara yardımcı olmak ve sorulara açık, anlaşılır cevaplar vermektir. Emin olmadığın konularda bunu belirtir, uydurma bilgi eklemezsin. Cevaplarını nazik, sade ve doğal bir Türkçe ile yazarsın."

# Questions to remove (Identity/Meta questions)
REMOVE_QUESTIONS = [
    "Senin adın ne?",
    "Neler yapabilirsin?",
    "Merhaba, nasılsın?"
]

def clean_and_update():
    cleaned_data = []
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                text = data['text']
                
                # Extract parts using regex for reliability
                sys_match = re.search(r"<system>(.*?)</system>", text, re.DOTALL)
                user_match = re.search(r"<user>(.*?)</user>", text, re.DOTALL)
                assist_match = re.search(r"<assistant>(.*?)</assistant>", text, re.DOTALL)
                
                if user_match and assist_match:
                    user_content = user_match.group(1).strip()
                    assist_content = assist_match.group(1).strip()
                    
                    # Filter logic
                    if user_content in REMOVE_QUESTIONS:
                        continue
                        
                    # Reconstruct with new system prompt
                    new_text = f"<system>{NEW_SYSTEM_PROMPT}</system>\n<user>{user_content}</user>\n<assistant>{assist_content}</assistant>"
                    cleaned_data.append({"text": new_text})

        # Write back
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for entry in cleaned_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                
        return len(cleaned_data)
        
    except Exception as e:
        print(f"Error: {e}")
        return 0

count = clean_and_update()
print(f"Dataset cleaned. Kept {count} examples with updated system prompt.")
