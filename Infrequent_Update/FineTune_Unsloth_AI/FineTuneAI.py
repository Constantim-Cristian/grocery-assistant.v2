# Simplified Product Quantity/Unit Extraction - FIXED VERSION
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
import torch
import json
from datasets import Dataset
import re
from datetime import datetime

torch.cuda.empty_cache()

max_seq_length = 1024
lora_rank = 32

# Load model - MUCH simpler model
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B",  # Simple, fast 3B model
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    dtype = torch.float16,
)

# Get chat template for Llama
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",  # Use llama chat format
)

# Fix tokenizer padding issues
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Add LoRA - same as before
model = FastModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = lora_rank,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# System prompt - simpler, no reasoning
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


# Load and process data - convert to chat format
def load_and_process_data():
    with open('homeworkData.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conversations = []
    for item in data:
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f" '{item['Title']}'"},
            {"role": "assistant", "content": f'{{"quantity": {item["Quantity"]}, "unit": "{item["Unit"]}"}}'}
        ]
        conversations.append({"conversations": conversation})
    
    return Dataset.from_list(conversations)

dataset = load_and_process_data()

# Apply chat template to conversations
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# Simple SFT Training - NO GRPO complexity
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,  # Higher batch size
        gradient_accumulation_steps = 2,  # Lower accumulation
        warmup_steps = 20,
        max_steps = 2500,  # More steps to prevent overfitting
        learning_rate = 2e-4,  # Lower learning rate
        logging_steps = 20,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none",
        output_dir = "outputs",
        save_steps = 400,
    ),
)

# Train only on assistant responses (ignore system/user)
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

print("Starting training...")
trainer_stats = trainer.train()

print(f"Training completed in {trainer_stats.metrics['train_runtime']} seconds")
print(f"That's {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")

# Prepare model for inference
model = FastModel.for_inference(model)

# Test the model with improved generation
test_cases = [
    "'G Dus M Whitwater 675Old Spice'",
    "'Cafetiera Actuel, 6 cesti, aluminiu, negru'", 
    "'Penar Neechipat Cu 2 Fermoare, Dimensiune 19,5 X 3 X 13,5 Cm'",
    "'Caiet capsat A4, 6 de file, velin'",
    "'Os Din Piele Presata Pentru Caini 15 Cm, 4Dog'",
    "'Set 4 markere permanente Auchan, varf mediu Set 4 markere permanente Auchan, varf mediu'"
]

print("\n=== TESTING THE MODEL ===")

# Store results for file output
test_results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "Llama-3.2-3B with LoRA fine-tuning",
    "training_steps": 400,
    "test_cases": []
}

for i, test_text in enumerate(test_cases):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": test_text},
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True,
    )
    
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    # Improved generation parameters - Fixed for newer Unsloth/Transformers
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 50,        # Just need JSON
            temperature = 0.3,          # Low temperature for consistency
            do_sample = False,          # Greedy decoding for clean output
            pad_token_id = tokenizer.pad_token_id,
            eos_token_id = tokenizer.eos_token_id,
            use_cache = False,          # Set to False to avoid cache issues
            cache_implementation = None,  # Disable cache implementation
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    
    # Clean the response - extract just the JSON
    json_match = re.search(r'\{"quantity".*?"unit".*?\}', response)
    if json_match:
        clean_response = json_match.group(0)
    else:
        # Fallback: take first line and clean it
        clean_response = response.split('\n')[0].strip()
        # Remove any remaining garbage characters
        clean_response = re.sub(r'[^\w\s{}":,.-]', '', clean_response)
    
    # Store result
    test_result = {
        "test_id": i + 1,
        "input": test_text,
        "output": clean_response,
        "raw_output": response[:200] + "..." if len(response) > 200 else response
    }
    test_results["test_cases"].append(test_result)
    
    # Print to console
    print(f"\nTest {i+1}:")
    print(f"Input: {test_text}")
    print(f"Clean Output: {clean_response}")
    if json_match is None:
        print(f"Raw Output (first 100 chars): {response[:100]}...")
    print("-" * 60)

# Save results to files
results_filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(results_filename, 'w', encoding='utf-8') as f:
    json.dump(test_results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Test results saved to: {results_filename}")

# Analysis
def analyze_results(results):
    analysis = {
        "total_tests": len(results["test_cases"]),
        "valid_json_count": 0,
        "valid_units_count": 0,
        "errors": []
    }
    
    valid_units = ["kg", "g", "ml", "l", "buc"]
    
    for result in results["test_cases"]:
        output = result["output"]
        
        # Check JSON format
        try:
            parsed = json.loads(output)
            if "quantity" in parsed and "unit" in parsed:
                analysis["valid_json_count"] += 1
            else:
                analysis["errors"].append(f"Test {result['test_id']}: Missing required JSON fields")
        except:
            analysis["errors"].append(f"Test {result['test_id']}: Invalid JSON format")
        
        # Check valid units
        if any(unit in output.lower() for unit in valid_units):
            analysis["valid_units_count"] += 1
        else:
            analysis["errors"].append(f"Test {result['test_id']}: Invalid or missing unit")
    
    return analysis

analysis = analyze_results(test_results)
analysis_filename = f"test_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

with open(analysis_filename, 'w', encoding='utf-8') as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)

print(f"✅ Analysis saved to: {analysis_filename}")
print(f"\nQuick Summary:")
print(f"- Valid JSON format: {analysis['valid_json_count']}/{analysis['total_tests']}")
print(f"- Valid units: {analysis['valid_units_count']}/{analysis['total_tests']}")
print(f"- Errors: {len(analysis['errors'])}")

# Save the model
model.save_pretrained("product_extraction_lora")
tokenizer.save_pretrained("product_extraction_lora")

print("✅ Model saved to 'product_extraction_lora'!")
print("✅ Training complete!")

# Optional: Show training loss progression
if hasattr(trainer_stats, 'log_history'):
    losses = [log['loss'] for log in trainer_stats.log_history if 'loss' in log]
    if losses:
        print(f"\nTraining Loss Progression:")
        print(f"Start: {losses[0]:.4f} → End: {losses[-1]:.4f}")
        print(f"Improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")