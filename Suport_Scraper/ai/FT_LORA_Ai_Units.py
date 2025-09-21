# JSON Batch Inference for Fine-tuned Product Extraction Model - In-Place Processing
from unsloth import FastModel
import torch
import json
import re
from datetime import datetime
import os
from tqdm import tqdm

# Clear GPU memory
torch.cuda.empty_cache()

# ==================================================
# CONFIGURATION
# ==================================================
MODEL_PATH = r"Suport_Scraper/ai/product_extraction_lora"  # Path to your saved model
BATCH_SIZE = 30                          # Batch size for GPU inference (tune to your GPU)

# System prompt (same as training)
system_prompt = """Extract quantity and unit from Romanian product title. Return ONLY JSON format: {"quantity": X, "unit": "Y"}.

RULES:
- Allowed units: kg, g, ml, l, buc
- Ignore dimensions, technical specs, model numbers (cm, inch, W, etc.)
- There are no negative values or quantaties.
- If there are two units with quantaties chose the second one.
- Default: {"quantity": 1, "unit": "buc"} if no unit found
- Handle multipacks (e.g., 4x0,33l) and promotions (e.g., 6+2 gratis)
- If a quantity is present but no unit:
  - For liquids (lapte, apa, suc, ulei, vin, bere, detergent lichid, sampon, gel):
    * Use ml if the number has 3 digits (e.g., 500 â†’ 500 ml, 6x300 â†’ 6x300 ml)
    * Use l if the number has 1â€“2 digits (e.g., 2 â†’ 2 l)
  - For powders/solids (faina, zahar, orez, sare, cafea macinata):
    * Use g if the number has 3 digits (e.g., 750 â†’ 750 g, 2x100 â†’ 2x100g)
    * Use kg if the number has 1â€“2 digits (e.g., 2 â†’ 2 kg)
  - Otherwise, default to {"quantity": N, "unit": "buc"}


Examples:
"Tigaie 20cm" -> {"quantity": 1, "unit": "buc"}
"Chips 4x333 g" -> {"quantity": "4x333", "unit": "g"}
"Costa D Oro Grezzo Ulei Masl Ev Nef 500" -> {"quantity": 500, "unit": "ml"}
"Spornic Clasic Ulei Floarea Soarelui 2" -> {"quantity": 2, "unit": "l"}
"G Dus M Whitwater 675Old Spice" -> {"quantity": 675, "unit": "ml"}
"Alabala Ultra Caserole Aluminiu 6 Portii 2/Set" -> {"quantity": 2, "unit": "buc"}
"Pachet 6 + 2 gratis lapte" -> {"quantity": "6+2", "unit": "buc"}"""

# Conversion configuration
CONVERSION_FACTORS = {
    "kg": ("g", 1000),
    "l": ("ml", 1000),
    "spalari": ("buc", 1),
    "bucati": ("buc", 1),
    "buc": ("buc", 1),
    "gr": ("g", 1),
    "ml": ("ml", 1),
    "g": ("g", 1),
}

# Global variables to store the loaded model
_global_model = None
_global_tokenizer = None

# ==================================================
# QUANTITY EVALUATION AND CONVERSION FUNCTIONS
# ==================================================
# ==================================================
# QUANTITY EVALUATION AND CONVERSION FUNCTIONS
# ==================================================
def evaluate_quantity(quant):
    """Safely evaluate a quantity string (e.g., "4*2" -> 8)"""
    try:
        # Handle string quantities like "6x10", "4*2", "6+2"
        if isinstance(quant, str):
            quant = quant.replace("x", "*").replace("X", "*").replace(",", ".")
            # Handle special cases like "6+2" (for promotions)
            if "+" in quant and "*" not in quant:
                # For promotions like "6+2", we might want to keep as string or calculate total
                # Let's calculate total for now
                parts = quant.split("+")
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    return int(parts[0]) + int(parts[1])
                else:
                    return quant  # Return as string if not simple addition
            else:
                return eval(quant)
        elif isinstance(quant, (int, float)):
            return quant
        else:
            return None
    except (SyntaxError, NameError, TypeError, ValueError):
        try:
            return float(quant)
        except (ValueError, TypeError):
            return quant  # Return original if cannot convert

def convert_to_smallest_unit(quant, unit):
    """Convert a quantity to the smallest unit"""
    if unit.lower() in CONVERSION_FACTORS:
        smallest_unit, factor = CONVERSION_FACTORS[unit.lower()]
        evaluated_quant = evaluate_quantity(quant)
        
        if isinstance(evaluated_quant, (int, float)):
            return evaluated_quant * factor, smallest_unit
        else:
            # If quantity is a complex string that couldn't be evaluated, return as is
            return quant, unit
    return quant, unit

def fix_malformed_json(json_string):
    """Fix malformed JSON by properly quoting quantity values"""
    try:
        # Try to parse first to see if it's valid
        json.loads(json_string)
        return json_string  # Already valid
    except json.JSONDecodeError:
        # Fix common issues: unquoted quantity values
        # Pattern to find unquoted quantity values (like 2x50, 8x125, etc.)
        pattern = r'"quantity":\s*([^,}\s]+)'
        
        def quote_match(match):
            quantity_value = match.group(1)
            # Check if it needs quoting (contains non-numeric characters)
            if not quantity_value.replace('.', '').replace('-', '').isdigit():
                return f'"quantity": "{quantity_value}"'
            return match.group(0)
        
        fixed_json = re.sub(pattern, quote_match, json_string)
        
        try:
            json.loads(fixed_json)
            return fixed_json
        except json.JSONDecodeError:
            # If still invalid, try more aggressive fixing
            try:
                # Extract quantity and unit using regex
                quantity_match = re.search(r'"quantity":\s*([^,}]+)', json_string)
                unit_match = re.search(r'"unit":\s*"([^"]+)"', json_string)
                
                if quantity_match and unit_match:
                    quantity = quantity_match.group(1).strip()
                    unit = unit_match.group(1)
                    
                    # Remove any trailing commas or invalid characters
                    quantity = re.sub(r'[,\}]', '', quantity).strip()
                    
                    # Create proper JSON
                    fixed_json = f'{{"quantity": "{quantity}", "unit": "{unit}"}}'
                    json.loads(fixed_json)  # Validate
                    return fixed_json
            except:
                pass
            
            # Last resort: return as string with error handling
            return json_string

def process_extracted_data(extracted_data):
    """Process the extracted data to evaluate quantities and convert to smallest units"""
    try:
        # First fix any malformed JSON
        fixed_data = fix_malformed_json(extracted_data)
        parsed_data = json.loads(fixed_data)
        quantity = parsed_data.get('quantity', 1)
        unit = parsed_data.get('unit', 'buc')
        
        # Convert to smallest unit
        converted_quantity, converted_unit = convert_to_smallest_unit(quantity, unit)
        
        # Create new parsed data with converted values
        processed_data = {
            'quantity': converted_quantity,
            'unit': converted_unit,
            'original_quantity': quantity,
            'original_unit': unit
        }
        
        return json.dumps(processed_data), processed_data
        
    except json.JSONDecodeError as e:
        print(f"âŒ Failed to parse JSON even after fixing: {extracted_data}")
        # Try to extract data using regex as fallback
        try:
            quantity_match = re.search(r'"quantity":\s*([^,}]+)', extracted_data)
            unit_match = re.search(r'"unit":\s*"([^"]+)"', extracted_data)
            
            if quantity_match and unit_match:
                quantity = quantity_match.group(1).strip().strip('"\'')
                unit = unit_match.group(1)
                
                # Convert to smallest unit
                converted_quantity, converted_unit = convert_to_smallest_unit(quantity, unit)
                
                processed_data = {
                    'quantity': converted_quantity,
                    'unit': converted_unit,
                    'original_quantity': quantity,
                    'original_unit': unit
                }
                
                return json.dumps(processed_data), processed_data
        except:
            pass
        
        return extracted_data, None

# ==================================================
# MODEL LOADING (CALLED FROM SCRAPER)
# ==================================================
def initialize_model():
    """Load the fine-tuned model and tokenizer - call this from your scraper"""
    global _global_model, _global_tokenizer
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at '{MODEL_PATH}'")
    
    print("ðŸ”„ Loading fine-tuned model...")

    _global_model, _global_tokenizer = FastModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=torch.float16,
    )

    # Prepare for inference
    _global_model = FastModel.for_inference(_global_model)

    # Ensure pad token and left padding
    if _global_tokenizer.pad_token is None:
        _global_tokenizer.pad_token = _global_tokenizer.eos_token
    _global_tokenizer.padding_side = "left"

    print(f"âœ… Model loaded and ready in memory (padding_side={_global_tokenizer.padding_side}).")

def get_loaded_model():
    """Get the pre-loaded model and tokenizer"""
    global _global_model, _global_tokenizer
    
    if _global_model is None or _global_tokenizer is None:
        raise RuntimeError("Model not loaded! Call initialize_model() first.")
    
    return _global_model, _global_tokenizer

# ==================================================
# DATA LOADING
# ==================================================
def load_json_data(file_path):
    """Load and validate input JSON data"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file '{file_path}' not found!")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"âœ… Loaded {len(data)} items from {file_path}")
    return data

def save_json_data(data, file_path):
    """Save JSON data back to file"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"âœ… Saved updated data to {file_path}")

# ==================================================
# RESPONSE CLEANUP
# ==================================================
def extract_json_from_response(response):
    """Clean and extract JSON from model response"""
    json_match = re.search(r'\{"quantity".*?"unit".*?\}', response)
    if json_match:
        return json_match.group(0)

    # Fallback: try first line
    clean_response = response.split("\n")[0].strip()
    clean_response = re.sub(r"[^\w\s{}\":,.\-]", "", clean_response)

    return clean_response

# ==================================================
# SINGLE ITEM INFERENCE
# ==================================================
def process_single_item(model, tokenizer, title, max_new_tokens=50):
    """Process a single product title"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"'{title}'"},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokenizer.padding_side = "left"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    # input length (padded length)
    input_len = inputs["input_ids"].size(1)
    gen_tokens = outputs[0][input_len:].cpu().tolist()
    response = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    clean_response = extract_json_from_response(response)

    return clean_response, response

# ==================================================
# BATCH INFERENCE (single forward pass)
# ==================================================
def process_batch_items(model, tokenizer, titles, max_new_tokens=50):
    """
    Process a batch of product titles with a single forward pass.
    Returns a list of (clean_output, raw_output) for each title.
    """
    messages_batch = []
    for title in titles:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"'{title}'"},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        messages_batch.append(text)

    # Tokenize batch with left-padding (we set tokenizer.padding_side = "left" earlier)
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        messages_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to("cuda")

    # Generate responses for the whole batch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    results = []
    input_len = inputs["input_ids"].size(1)  # padded input length (same for all in batch)
    for i in range(len(titles)):
        gen_tokens = outputs[i][input_len:].cpu().tolist()
        response = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        clean_response = extract_json_from_response(response)
        results.append((clean_response, response))

    return results

# ==================================================
# BATCH WITH OOM-RESILIENT RETRY
# ==================================================
def process_batch_with_retry(model, tokenizer, titles, max_new_tokens=50):
    """
    Try to process the whole batch; on CUDA OOM, split recursively until success.
    Returns concatenated results for all titles (order-preserving).
    """
    try:
        return process_batch_items(model, tokenizer, titles, max_new_tokens=max_new_tokens)
    except RuntimeError as e:
        err = str(e).lower()
        if "out of memory" in err or "cuda out of memory" in err or "oom" in err:
            torch.cuda.empty_cache()
            if len(titles) <= 1:
                # give up: let caller fallback to single-item processing
                raise
            mid = len(titles) // 2
            left = process_batch_with_retry(model, tokenizer, titles[:mid], max_new_tokens=max_new_tokens)
            right = process_batch_with_retry(model, tokenizer, titles[mid:], max_new_tokens=max_new_tokens)
            return left + right
        else:
            # unknown error: bubble up
            raise

# ==================================================
# VALIDATION
# ==================================================
def validate_output(output):
    """Validate the model output"""
    try:
        parsed = json.loads(output)
        if "quantity" not in parsed or "unit" not in parsed:
            return False, "Missing required fields"
        valid_units = ["kg", "g", "ml", "l", "buc"]
        if parsed["unit"] not in valid_units:
            return False, f"Invalid unit: {parsed['unit']}"
        return True, "Valid"
    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {str(e)}"

# ==================================================
# IN-PLACE PROCESSING FUNCTION
# ==================================================
def process_rerun_items_inplace(input_file):
    """Process items with Rerun_Flag=true and update them in-place in the original file"""
    print(f"ðŸš€ Starting in-place processing of items with Rerun_Flag=true...")
    print(f"File: {input_file}")
    print("-" * 50)

    # Use pre-loaded model
    model, tokenizer = get_loaded_model()
    
    # Load data
    input_data = load_json_data(input_file)
    
    # Filter items that need reprocessing
    items_to_process = []
    item_indices = []  # Track original positions
    
    for i, item in enumerate(input_data):
        if isinstance(item, dict) and (item.get('Rerun_Flag') == True or item.get('LowValFlag')== "Low"):
            items_to_process.append(item)
            item_indices.append(i)
    
    if not items_to_process:
        print("â„¹ï¸ No items found with Rerun_Flag=true")
        return {"processed": 0, "successful": 0, "failed": 0}
    
    print(f"ðŸ“‹ Found {len(items_to_process)} items to reprocess")
    
    # Extract titles from items to process
    titles = []
    for item in items_to_process:
        title = next((item.get(f) for f in ["Title", "title", "name", "product_name"] if f in item), "")
        titles.append(title)
    
    # Statistics
    stats = {"processed": 0, "successful": 0, "failed": 0}
    
    # Process in batches
    for batch_start in tqdm(range(0, len(items_to_process), BATCH_SIZE)):
        batch_end = min(batch_start + BATCH_SIZE, len(items_to_process))
        batch_items = items_to_process[batch_start:batch_end]
        batch_titles = titles[batch_start:batch_end]
        batch_indices = item_indices[batch_start:batch_end]

        try:
            # Use OOM-resilient batch processing
            batch_results = process_batch_with_retry(model, tokenizer, batch_titles)

            for i, ((clean_output, raw_output), original_index) in enumerate(zip(batch_results, batch_indices)):
                stats["processed"] += 1
                
                try:
                    # Process the extracted data (evaluate quantities and convert to smallest units)
                    processed_output, processed_data = process_extracted_data(clean_output)
                    
                    if processed_data:
                        # Update the original item in input_data
                        input_data[original_index]['Quantity'] = processed_data['quantity']
                        input_data[original_index]['Unit'] = processed_data['unit']
                        input_data[original_index]['original_quantity'] = processed_data['original_quantity']
                        input_data[original_index]['original_unit'] = processed_data['original_unit']
                        input_data[original_index]['extracted_data'] = processed_output
                        input_data[original_index]['Rerun_Flag'] = False
                        input_data[original_index]['last_processed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        stats["successful"] += 1
                    else:
                        # If processing failed, use the original extracted data
                        parsed_data = json.loads(clean_output)
                        input_data[original_index]['Quantity'] = parsed_data['quantity']
                        input_data[original_index]['Unit'] = parsed_data['unit']
                        input_data[original_index]['extracted_data'] = clean_output
                        input_data[original_index]['Rerun_Flag'] = False
                        input_data[original_index]['last_processed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        stats["successful"] += 1
                    
                except json.JSONDecodeError as e:
                    print(f"âŒ Failed to parse JSON for item {original_index}: {clean_output}")
                    # Keep Rerun_Flag=true for failed items
                    input_data[original_index]['processing_error'] = f"JSON parse error: {str(e)}"
                    input_data[original_index]['last_processed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    stats["failed"] += 1
                    
                except Exception as e:
                    print(f"âŒ Error updating item {original_index}: {str(e)}")
                    input_data[original_index]['processing_error'] = str(e)
                    input_data[original_index]['last_processed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    stats["failed"] += 1

        except Exception as e:
            # Batch-level failure - fall back to single-item processing
            print(f"âŒ Error processing batch {batch_start}-{batch_end}: {str(e)}")
            print("ðŸ‘‰ Falling back to single-item processing for this batch")

            for i, (title, original_index) in enumerate(zip(batch_titles, batch_indices)):
                stats["processed"] += 1
                
                try:
                    clean_output, raw_output = process_single_item(model, tokenizer, title)
                    
                    # Process the extracted data (evaluate quantities and convert to smallest units)
                    processed_output, processed_data = process_extracted_data(clean_output)
                    
                    if processed_data:
                        # Update the original item in input_data
                        input_data[original_index]['Quantity'] = processed_data['quantity']
                        input_data[original_index]['Unit'] = processed_data['unit']
                        input_data[original_index]['original_quantity'] = processed_data['original_quantity']
                        input_data[original_index]['original_unit'] = processed_data['original_unit']
                        input_data[original_index]['extracted_data'] = processed_output
                        input_data[original_index]['Rerun_Flag'] = False
                        input_data[original_index]['last_processed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        # If processing failed, use the original extracted data
                        parsed_data = json.loads(clean_output)
                        input_data[original_index]['Quantity'] = parsed_data['quantity']
                        input_data[original_index]['Unit'] = parsed_data['unit']
                        input_data[original_index]['extracted_data'] = clean_output
                        input_data[original_index]['Rerun_Flag'] = False
                        input_data[original_index]['last_processed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    stats["successful"] += 1
                    
                except json.JSONDecodeError as e:
                    print(f"âŒ Failed to parse JSON for item {original_index}: {clean_output}")
                    input_data[original_index]['processing_error'] = f"JSON parse error: {str(e)}"
                    input_data[original_index]['last_processed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    stats["failed"] += 1
                    
                except Exception as inner_e:
                    print(f"âŒ Error processing item {original_index}: {str(inner_e)}")
                    input_data[original_index]['processing_error'] = str(inner_e)
                    input_data[original_index]['last_processed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    stats["failed"] += 1

    # Save updated data back to the original file
    print(f"\nðŸ’¾ Saving updated data back to {input_file}...")
    save_json_data(input_data, input_file)

    # ==================================================
    # APPEND TO processed_quant_units.json
    # ==================================================
    processed_file = r"Suport_Scraper/AI_Proccesed_Units.json"
    new_results = []

    # Build result_item objects for successful updates
    for idx in item_indices:
        item = input_data[idx]
        if item.get("Rerun_Flag") is False and "Quantity" in item and "Unit" in item:
            title = next((item.get(f) for f in ["Title", "title", "name", "product_name"] if f in item), "")
            clean_output = item.get("extracted_data", "")
            raw_output = item.get("extracted_data", "")
            is_valid, validation_msg = validate_output(clean_output)

            result_item = {
                "id": idx + 1,
                "original_item": item,
                "title": title,
                "extracted_data": clean_output,
                "raw_output": raw_output[:200] + "..." if len(raw_output) > 200 else raw_output,
                "is_valid": is_valid,
                "validation_message": validation_msg,
                "quantity": item.get("Quantity"),  # Add lowercase for AI_Processed_Units
                "unit": item.get("Unit"),
            }
            new_results.append(result_item)

    # Load existing file if available
    if os.path.exists(processed_file):
        with open(processed_file, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
                existing_results = existing_data.get("processed_items", []) if isinstance(existing_data, dict) else existing_data
            except json.JSONDecodeError:
                existing_results = []
    else:
        existing_results = []

    # Deduplicate by title
    existing_titles = {r["title"] for r in existing_results if "title" in r}
    merged_results = existing_results + [r for r in new_results if r["title"] not in existing_titles]

    # Save back
    with open(processed_file, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, indent=2, ensure_ascii=False)

    print(f"âœ… Appended {len(new_results)} new results to {processed_file}")

    # Print summary
    print("\n" + "=" * 50)
    print("ðŸ“Š PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Items processed: {stats['processed']}")
    print(f"Successfully updated: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success rate: {(stats['successful'] / stats['processed'] * 100):.1f}%" if stats['processed'] > 0 else "0%")
    print(f"\nâœ… File updated in-place: {input_file}")

    return stats

# ==================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# ==================================================
def process_json_batch_with_preloaded_model(input_file, output_file=None):
    """Process JSON batch using pre-loaded model - creates new file (backward compatibility)"""
    if output_file is None:
        output_file = r"Suport_Scraper/AI_Proccesed_Units.json"
    
    print(f"ðŸš€ Starting batch processing with pre-loaded model...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print("-" * 50)

    # Use pre-loaded model
    model, tokenizer = get_loaded_model()
    input_data = load_json_data(input_file)

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": input_file,
        "total_items": len(input_data),
        "processed_items": [],
        "statistics": {"successful": 0, "failed": 0, "valid_json": 0, "invalid_json": 0},
    }

    # Extract titles
    titles = []
    for item in input_data:
        if isinstance(item, dict):
            title = next((item.get(f) for f in ["Title", "title", "name", "product_name"] if f in item), "")
        elif isinstance(item, str):
            title = item
        else:
            title = str(item)
        titles.append(title)

    # Process in batches
    for batch_start in tqdm(range(0, len(input_data), BATCH_SIZE)):
        batch_end = min(batch_start + BATCH_SIZE, len(input_data))
        batch_items = input_data[batch_start:batch_end]
        batch_titles = titles[batch_start:batch_end]

        try:
            # Use OOM-resilient batch processing
            batch_results = process_batch_with_retry(model, tokenizer, batch_titles)

            for i, (item, title, (clean_output, raw_output)) in enumerate(zip(batch_items, batch_titles, batch_results)):
                item_id = batch_start + i + 1

                # Process the extracted data (evaluate quantities and convert to smallest units)
                processed_output, processed_data = process_extracted_data(clean_output)
                
                if processed_data:
                    clean_output = processed_output
                    parsed_data = processed_data
                else:
                    try:
                        parsed_data = json.loads(clean_output)
                    except:
                        parsed_data = None

                is_valid, validation_msg = validate_output(clean_output)
                result_item = {
                    "id": item_id,
                    "original_item": item,
                    "title": title,
                    "extracted_data": clean_output,
                    "raw_output": raw_output[:200] + "..." if len(raw_output) > 200 else raw_output,
                    "is_valid": is_valid,
                    "validation_message": validation_msg,
                    "parsed_data": parsed_data,
                }

                results["processed_items"].append(result_item)
                results["statistics"]["successful"] += 1
                if is_valid:
                    results["statistics"]["valid_json"] += 1
                else:
                    results["statistics"]["invalid_json"] += 1

        except Exception as e:
            # Batch-level failure (non-OOM or final fallback)
            print(f"âŒ Error processing batch {batch_start}-{batch_end}: {str(e)}")
            print("ðŸ‘‰ Falling back to single-item processing for this batch")

            for i, (item, title) in enumerate(zip(batch_items, batch_titles)):
                item_id = batch_start + i + 1
                try:
                    clean_output, raw_output = process_single_item(model, tokenizer, title)
                    
                    # Process the extracted data (evaluate quantities and convert to smallest units)
                    processed_output, processed_data = process_extracted_data(clean_output)
                    
                    if processed_data:
                        clean_output = processed_output
                        parsed_data = processed_data
                    else:
                        try:
                            parsed_data = json.loads(clean_output)
                        except:
                            parsed_data = None

                    is_valid, validation_msg = validate_output(clean_output)

                    result_item = {
                        "id": item_id,
                        "original_item": item,
                        "title": title,
                        "extracted_data": clean_output,
                        "raw_output": raw_output[:200] + "..." if len(raw_output) > 200 else raw_output,
                        "is_valid": is_valid,
                        "validation_message": validation_msg,
                        "parsed_data": parsed_data,
                    }

                    results["processed_items"].append(result_item)
                    results["statistics"]["successful"] += 1
                    if is_valid:
                        results["statistics"]["valid_json"] += 1
                    else:
                        results["statistics"]["invalid_json"] += 1

                except Exception as inner_e:
                    print(f"âŒ Error processing item {item_id}: {str(inner_e)}")
                    results["statistics"]["failed"] += 1
                    results["processed_items"].append(
                        {
                            "id": item_id,
                            "original_item": item,
                            "title": title,
                            "error": str(inner_e),
                            "is_valid": False,
                            "validation_message": "Processing error",
                        }
                    )

    # Save results
    print(f"\nðŸ’¾ Saving results to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results["processed_items"], f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 50)
    print("ðŸ“Š PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Total items: {results['statistics']['successful'] + results['statistics']['failed']}")
    print(f"Successfully processed: {results['statistics']['successful']}")
    print(f"Failed: {results['statistics']['failed']}")
    print(f"Valid JSON outputs: {results['statistics']['valid_json']}")
    print(f"Invalid JSON outputs: {results['statistics']['invalid_json']}")
    print(f"Success rate: {(results['statistics']['valid_json'] / results['total_items'] * 100):.1f}%")
    print(f"\nâœ… Results saved to: {output_file}")

    return results

def load_model():
    """Original model loading function - for backward compatibility"""
    print("Loading fine-tuned model...")

    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=torch.float16,
    )

    # Prepare for inference
    model = FastModel.for_inference(model)

    # Ensure pad token and left padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"âœ… Model and tokenizer ready (padding_side={tokenizer.padding_side}).")
    return model, tokenizer

# ==================================================
# SAMPLE DATA CREATOR
# ==================================================
def create_sample_input_with_rerun_flags():
    sample_data = [
        {
            "Title": "G Dus M Whitwater 675Old Spice",
            "Rerun_Flag": True,
            "price": "15.99"
        },
        {
            "Title": "Cafetiera Actuel, 6 cesti, aluminiu, negru",
            "Rerun_Flag": False,
            "quantity": 1,
            "unit": "buc",
            "price": "89.99"
        },
        {
            "Title": "Lapte UHT 2.5% 1l Zuzu",
            "Rerun_Flag": True,
            "price": "4.50"
        },
        {
            "Title": "Orez basmati 500g",
            "Rerun_Flag": True,
            "price": "12.30"
        },
        {
            "Title": "Caiet capsat A4, 60 de file, velin",
            "Rerun_Flag": False,
            "quantity": 1,
            "unit": "buc",
            "price": "3.20"
        }
    ]

    with open("sample_input_with_flags.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)

    print("âœ… Sample input file with Rerun_Flag created: sample_input_with_flags.json")

# ==================================================
# ENTRY POINT
# ==================================================
if __name__ == "__main__":
    today = datetime.now()
    date_str = today.strftime("%d-%m-%Y")
    INPUT_JSON_FILE = rf"Scraped_Data/products/products_{date_str}.json"
    
    if not os.path.exists(INPUT_JSON_FILE):
        print(f"âš ï¸ Input file '{INPUT_JSON_FILE}' not found. Creating sample input...")
        create_sample_input_with_rerun_flags()
        INPUT_JSON_FILE = "sample_input_with_flags.json"

    try:
        # For standalone usage - load model and process in-place
        initialize_model()
        results = process_rerun_items_inplace(INPUT_JSON_FILE)
        print(f"\nðŸŽ‰ Processing completed! Updated {results['successful']} items in-place.")
    except Exception as e:
        print(f"âŒ Processing failed: {str(e)}")
        import traceback
        traceback.print_exc()