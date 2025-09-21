import json
import random
import re
import requests
import concurrent.futures

OLLAMA_URL = "http://localhost:11434/api/generate"

def query_ollama_http(model, prompt):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()
    return resp.json()["response"]

def build_prompt(title: str) -> str:
    return f"""
Extract quantity and unit from Romanian product title. Return ONLY JSON.

STRICT RULES:
- Allowed units: "kg", "g", "ml", "l", "buc".
- If no valid unit is found, default to {{"quantity": 1, "unit": "buc"}}.
- Ignore dimensions (cm, mm, m), technical specs (MBar, volts, watts, bar), model numbers, clothing sizes, pot sizes.
- Prefer weight/volume patterns first, then counts (portii, set, bucati).
- If the title contains a multipack pattern like 'AxB' (e.g., '4x0,33 l'), return both numbers in the JSON:
    "quantity": A,
    "unit": the unit from B (kg, g, l, ml, buc)
- If the title contains promotional patterns like '6 + 2 free', return the total quantity (e.g., 8) and the unit (usually 'buc').
- Only extract numbers that refer to actual product quantity or volume/weight.
- Ignore model numbers, product codes, dimensions, technical specs, or other unrelated numbers.

Output example:
"Extract from: 'Lapte Zuzu 500ml'" -> {{"quantity": 500, "unit": "ml"}}
"Extract from: 'Paine alba 1kg'" -> {{"quantity": 1, "unit": "kg"}}
"Extract from: 'Tigaie 20cm'" -> {{"quantity": 1, "unit": "buc"}}
"Extract from: 'Alabala Ultra Caserole Aluminiu 6 Portii 2/Set'" -> {{"quantity": 2, "unit": "buc"}}
"Extract from: 'Jucarie de plus maimuta'" -> {{"quantity": 1, "unit": "buc"}}
"Extract from: 'Suc de portocale cu pulpa 2l'" -> {{"quantity": 2, "unit": "l"}}
"Extract from: 'Coca-Cola Zero 4x0,33 l'" -> {{"quantity": "4x0,33", "unit": "l"}}
"Extract from: 'Pachet 6 + 2 gratis lapte'" -> {{"quantity": "6+2", "unit": "buc"}}

Now extract from: '{title}' -> JSON:
"""

def process_item(item, model):
    title = item.get("Title", "")
    prompt = build_prompt(title)
    try:
        raw_output = query_ollama_http(model, prompt).strip()

        # Extract JSON safely
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in model output")

        extracted_info = json.loads(match.group(0))
        quantity = extracted_info.get('quantity', 1)
        unit = extracted_info.get('unit', 'buc')

    except Exception as e:
        print(f"❌ Could not process '{title}'. Error: {e}")
        quantity = 1
        unit = 'buc'

    # Return only the fields you want
    return {
        "Title": title,
        "Quantity": quantity,
        "Unit": unit
    }

def process_product_data(input_file, output_file, model_name="llama3", num_items=2000, max_workers=4):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: The file '{input_file}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: The file '{input_file}' is not valid JSON.")
        return

    if len(data) < num_items:
        print(f"⚠️ File has fewer than {num_items} items. Processing all {len(data)} items.")
        items_to_process = data
    else:
        items_to_process = random.sample(data, num_items)

    processed_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_item, item, model_name) for item in items_to_process]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            processed_item = future.result()
            processed_data.append(processed_item)
            print(f"✅ Processed {i + 1}/{len(items_to_process)}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print(f"🎉 Processing complete! Saved updated data to '{output_file}'.")


# --- Usage ---
# Make sure you pulled the model first:
#   ollama pull gpt-oss:20b
process_product_data(
    input_file='testdump.json',
    output_file='homeworkData.json',
    model_name='gpt-oss:20b',
    max_workers=4,
    num_items=1800  # try 2–4 depending on your CPU/GPU
)   
