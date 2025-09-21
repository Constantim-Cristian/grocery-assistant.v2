#!/usr/bin/env python3
"""
Terminal chat to test your trained Qwen model
"""
import torch
import json
import re
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

# Load your trained model
print("🔄 Loading your trained model...")
model, tokenizer = FastModel.from_pretrained(
    "product_extraction_lora",  # Your saved model path
    max_seq_length = 1024,
    load_in_4bit = True,
    dtype = torch.float16,
)

# Prepare for inference
model = FastModel.for_inference(model)
print("✅ Model loaded successfully!")

# System prompt (same as training)
system_prompt = """Extract quantity and unit from Romanian product title. Return ONLY JSON format: {"quantity": X, "unit": "Y"}.

RULES:
- Allowed units: kg, g, ml, l, buc
- Ignore dimensions, technical specs, model numbers (cm, inch, W, etc.)
- Default: {"quantity": 1, "unit": "buc"} if no unit found
- Handle multipacks (e.g., 4x0,33l) and promotions (e.g., 6+2 gratis)
- If a quantity is present but no unit:
  - For liquids (lapte, apa, suc, ulei, vin, bere, detergent lichid, sampon, gel):
    * Use ml if the number has 3 digits (e.g., 500 → 500 ml)
    * Use l if the number has 1–2 digits (e.g., 2 → 2 l)
  - For powders/solids (faina, zahar, orez, sare, cafea macinata):
    * Use g if the number has 3 digits (e.g., 750 → 750 g)
    * Use kg if the number has 1–2 digits (e.g., 2 → 2 kg)
  - Otherwise, default to {"quantity": N, "unit": "buc"}

Examples:
"Tigaie 20cm" -> {"quantity": 1, "unit": "buc"}
"Chips 4x0,33 g" -> {"quantity": "4x0,33", "unit": "g"}
"Costa D Oro Grezzo Ulei Masl Ev Nef 500" -> {"quantity": 500, "unit": "ml"}
"Spornic Clasic Ulei Floarea Soarelui 2" -> {"quantity": 2, "unit": "l"}
"G Dus M Whitwater 675Old Spice" -> {"quantity": 675, "unit": "ml"}
"Alabala Ultra Caserole Aluminiu 6 Portii 2/Set" -> {"quantity": 2, "unit": "buc"}
"Pachet 6 + 2 gratis lapte" -> {"quantity": "6+2", "unit": "buc"}

"""

def generate_response(product_title):
    """Generate response from your trained model"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Extract from: '{product_title}'"},
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True,
    )
    
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 50,
            temperature = 0.1,
            do_sample = True,
            top_p = 0.9,
            pad_token_id = tokenizer.pad_token_id,
            eos_token_id = tokenizer.eos_token_id,
            use_cache = False,          # Fixed to avoid StaticCache error
            cache_implementation = None,
        )
    
    response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    
    # Clean the response - extract JSON
    json_match = re.search(r'\{"quantity".*?"unit".*?\}', response)
    if json_match:
        clean_response = json_match.group(0)
    else:
        clean_response = response.split('\n')[0].strip()
        clean_response = re.sub(r'[^\w\s{}":,.-]', '', clean_response)
    
    return clean_response, response

def print_header():
    print("=" * 60)
    print("🤖 YOUR TRAINED QWEN MODEL - PRODUCT EXTRACTOR")
    print("=" * 60)
    print("📋 Model: Qwen3-4B-Instruct + LoRA Fine-tuning")
    print("📊 Training Loss: 0.3643 → 0.0138 (96% improvement!)")
    print("⌨️  Commands: 'exit' to quit, 'clear' to clear screen")
    print("🧪 Test Examples: Try the products from your training data")
    print("-" * 60)

def main():
    print_header()
    
    # Quick test examples
    examples = [
        "G Dus M Whitwater 675Old Spice",
        "Coca-Cola 4x0,33 l",
        "Tigaie cu strat antiaderent 28 cm",
        "Cafea Lavazza 250g",
        "Invat cu Zigaloo -Numere pana la 100"
    ]
    
    print("🚀 Quick test examples (just press Enter to skip):")
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example}")
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("🔍 Enter product title (or example number): ").strip()
            
            # Handle commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            elif user_input.lower() in ['clear', 'cls']:
                import os
                os.system('cls' if os.name == 'nt' else 'clear')
                print_header()
                continue
            elif user_input.isdigit() and 1 <= int(user_input) <= len(examples):
                user_input = examples[int(user_input) - 1]
                print(f"📝 Selected: '{user_input}'")
            elif not user_input:
                continue
            
            # Generate response from your model
            print("⚙️  Your trained model is thinking...", end="", flush=True)
            
            try:
                clean_result, raw_response = generate_response(user_input)
                
                print("\r✅ Model Response:")
                print("-" * 50)
                print("🎯 CLEAN OUTPUT:")
                print(clean_result)
                
                # Try to parse as JSON for validation
                try:
                    parsed = json.loads(clean_result)
                    print("\n✅ Valid JSON format!")
                    print(f"   Quantity: {parsed.get('quantity', 'Missing')}")
                    print(f"   Unit: {parsed.get('unit', 'Missing')}")
                except json.JSONDecodeError:
                    print("\n⚠️  Warning: Output is not valid JSON")
                
                # Show raw output if different
                if clean_result.strip() != raw_response.strip():
                    print(f"\n🔍 RAW OUTPUT (first 100 chars):")
                    print(f"   {raw_response[:100]}{'...' if len(raw_response) > 100 else ''}")
                
                print("-" * 50)
                
            except Exception as e:
                print(f"\n❌ Model Error: {e}")
                print("💡 This might be due to CUDA memory or generation issues")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    # Check if model exists
    import os
    if not os.path.exists("product_extraction_qwen"):
        print("❌ Error: Model not found!")
        print("💡 Make sure you've run the training script first and the model is saved to 'product_extraction_qwen'")
        exit(1)
    
    main()