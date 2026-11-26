import torch

def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    generator=None,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> str:
    """
    Generate text using a trained GPT model.
    
    Args:
        model: The trained GPT model.
        tokenizer: Tokenizer instance.
        prompt: Initial string to start generation.
        max_new_tokens: Number of tokens to generate.
        temperature: Scaling factor for logits. Must be > 0.
        generator: Optional random generator for reproducible sampling.
        top_k: Optional top-k filtering (0 disables).
        top_p: Optional nucleus (top-p) filtering (1.0 disables).
        repetition_penalty: Penalty (> 1.0) on previously generated tokens.
        
    Returns:
        The complete generated string (prompt + generated tokens).
    """
    if temperature <= 0.0:
        raise ValueError("Temperature must be greater than 0")
        
    model.eval()
    device = next(model.parameters()).device
    
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    
    max_context = model.max_context_length
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Truncate to context window
            context = input_ids[:, -max_context:]
            
            # Forward pass
            logits = model(context)
            
            # Take logits for the last token in the sequence
            last_logits = logits[:, -1, :].clone()

            # Apply repetition penalty to seen tokens in context
            if repetition_penalty > 1.0:
                seen_tokens = torch.unique(context[0])
                seen_scores = last_logits[0, seen_tokens]
                last_logits[0, seen_tokens] = torch.where(
                    seen_scores < 0,
                    seen_scores * repetition_penalty,
                    seen_scores / repetition_penalty,
                )

            # Apply temperature
            scaled_logits = last_logits / temperature

            # Top-k filtering
            if top_k > 0 and top_k < scaled_logits.size(-1):
                v, _ = torch.topk(scaled_logits, top_k)
                scaled_logits = scaled_logits.masked_fill(scaled_logits < v[:, [-1]], float("-inf"))

            # Top-p (nucleus) filtering
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                scaled_logits = scaled_logits.masked_fill(indices_to_remove, float("-inf"))
            
            # Convert to probabilities
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
            
            # Append token to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
    # Decode complete sequence
    output_ids = input_ids[0].tolist()
    return tokenizer.decode(output_ids)
