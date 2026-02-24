
import re

def clean_file(input_filename, output_filename):
    print(f"Cleaning {input_filename} -> {output_filename}...")
    
    with open(input_filename, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        
    print(f"Original line count: {len(lines)}")
    
    cleaned_lines = []
    
    # Pre-compile regex for splitting merged lines (4 or more spaces)
    # Using a positive lookahead/behind or just split and non-empty filter
    split_pattern = re.compile(r'\s{4,}')
    
    seen_lines = set()

    for line in lines:
        line = line.strip()
        
        # 1. Skip empty lines
        if not line:
            continue
            
        # 2. Skip headers (lines starting with #)
        if line.startswith('#'):
            continue
            
        # 3. Handle merged lines by splitting on large whitespace gaps
        # This handles cases like "sentence one.    sentence two."
        parts = split_pattern.split(line)
        
        for part in parts:
            part = part.strip()
            
            # Skip empty parts derived from split
            if not part:
                continue
            
            # 5. Normalize whitespace (collapse multiple spaces to one)
            part = re.sub(r'\s+', ' ', part)
            
            # 4. Deduplication
            if part in seen_lines:
                continue
            
            seen_lines.add(part)
            cleaned_lines.append(part)
            
    # Write to output file
    with open(output_filename, 'w', encoding='utf-8') as f:
        for line in cleaned_lines:
            f.write(line + '\n')
            
    print(f"Cleaned line count: {len(cleaned_lines)}")
    print("Done.")

if __name__ == "__main__":
    clean_file("foundation_corpus.txt", "foundation_corpus_clean.txt")
