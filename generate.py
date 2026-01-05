"""Text generation script"""

import argparse
from unsloth import FastLanguageModel
import config


def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    """Generate response for a prompt"""
    FastLanguageModel.for_inference(model)
    
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def main():
    parser = argparse.ArgumentParser(description='Generate text with fine-tuned model')
    parser.add_argument('--prompt', type=str, required=True, help='Input prompt')
    parser.add_argument('--max-tokens', type=int, default=256, help='Max new tokens')
    args = parser.parse_args()
    
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.FINAL_MODEL_DIR,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    
    print(f"\nPrompt: {args.prompt}\n")
    response = generate_response(model, tokenizer, args.prompt, args.max_tokens)
    print(f"Response: {response}\n")


if __name__ == "__main__":
    main()