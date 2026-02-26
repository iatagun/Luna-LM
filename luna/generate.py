"""
Luna-LM Metin Üretimi
Temperature + Top-K + Top-P + Repetition Penalty
Projedeki TEK generate_text kaynağı.
"""

import torch


def generate_text(model, tokenizer, device, start_text, max_new_tokens=100,
                  temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.2):
    """
    Metin üretimi (Temperature + Top-K + Top-P + Repetition Penalty)
    
    Args:
        model: GPTModel instance
        tokenizer: encode/decode metotları olan tokenizer
        device: torch device
        start_text: Başlangıç metni
        max_new_tokens: Üretilecek maksimum token sayısı
        temperature: Sampling sıcaklığı (0=greedy, düşük=deterministik, yüksek=yaratıcı)
        top_k: Top-k sampling (0=disable)
        top_p: Top-p (nucleus) sampling (1.0=disable)
        repetition_penalty: Tekrar cezası (1.0=disable)
    """
    model.eval()
    
    encoded = tokenizer.encode(start_text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_size = model.pos_emb.weight.shape[0]
            idx_cond = encoded_tensor[:, -context_size:]
            
            logits = model(idx_cond)
            logits = logits[:, -1, :]
            
            # 1. Repetition Penalty
            if repetition_penalty != 1.0:
                for i in range(encoded_tensor.size(0)):
                    for token_id in set(encoded_tensor[i].tolist()):
                        if logits[i, token_id] < 0:
                            logits[i, token_id] *= repetition_penalty
                        else:
                            logits[i, token_id] /= repetition_penalty

            # 2. Temperature
            if temperature > 0:
                logits = logits / temperature
            
            # 3. Top-K Filtering
            if top_k > 0:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                min_value = top_values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_value, 
                                    torch.full_like(logits, float('-inf')), 
                                    logits)
            
            # 4. Top-P (Nucleus) Filtering
            if top_p > 0 and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample
            if temperature > 0:
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
            encoded_tensor = torch.cat((encoded_tensor, idx_next), dim=1)
    
    output_ids = encoded_tensor.squeeze(0).tolist()
    decoded = tokenizer.decode(output_ids)
    return decoded
