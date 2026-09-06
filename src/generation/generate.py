import torch

def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float = 1.0, generator=None) -> str:
    """
    Generate text using a trained GPT model.
    
    Args:
        model: The trained GPT model.
        tokenizer: CharacterTokenizer instance.
        prompt: Initial string to start generation.
        max_new_tokens: Number of tokens to generate.
        temperature: Scaling factor for logits. Must be > 0.
        generator: Optional random generator for reproducible sampling.
        
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
            last_logits = logits[:, -1, :]
            
            # Apply temperature
            scaled_logits = last_logits / temperature
            
            # Convert to probabilities
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
            
            # Append token to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
    # Decode complete sequence
    output_ids = input_ids[0].tolist()
    return tokenizer.decode(output_ids)
