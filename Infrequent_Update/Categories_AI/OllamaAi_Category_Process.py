import json
import requests

def call_ollama(prompt):
    """Call Ollama API"""
    try:
        response = requests.post("http://localhost:11434/api/generate", 
            json={"model": "gpt-oss:20b", "prompt": prompt, "stream": False}, 
            timeout=300)
        response.raise_for_status()
        result = response.json()["response"].strip()
        if not result:
            print("Warning: Empty response from Ollama")
            return None
        return result
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None

def match_categories(main_file, comparison_file, output_file):
    """Match categories using AI"""
    
    # Load files
    with open(main_file, 'r', encoding='utf-8') as f:
        main_data = json.load(f)
    with open(comparison_file, 'r', encoding='utf-8') as f:
        comparison_data = json.load(f)
    
    # Extract unique categories from main file
    categories = []
    seen = set()
    for item in main_data:
        if 'CategorySlug' in item and 'MapMainSlug' in item:
            pair = (item['CategorySlug'], item['MapMainSlug'])
            if pair not in seen:
                categories.append({'CategorySlug': item['CategorySlug'], 'MapMainSlug': item['MapMainSlug']})
                seen.add(pair)
    
    # Extract comparison categories from the "categories" array
    comparison_categories = comparison_data.get("categories", [])
    
    if not comparison_categories:
        print("Error: No 'categories' array found in comparison file!")
        return
    
    print(f"Found {len(categories)} categories to match")
    print(f"Found {len(comparison_categories)} comparison categories")
    print("Available categories to match to:")
    for i, cat in enumerate(comparison_categories[:10], 1):
        print(f"  {i}. {cat}")
    if len(comparison_categories) > 10:
        print(f"  ... and {len(comparison_categories) - 10} more")
    
    # Process in batches of 5 (smaller batches for more reliable JSON)
    matches = []
    batch_size = 5
    
    for i in range(0, len(categories), batch_size):
        batch = categories[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(categories)-1)//batch_size + 1}...")
        
        # Format categories for prompt
        main_text = "\n".join([f"CategorySlug: '{cat['CategorySlug']}', MapMainSlug: '{cat['MapMainSlug']}'" 
                              for cat in batch])
        comp_text = "\n".join([f"- {cat}" for cat in comparison_categories])
        
        prompt = f"""Match each category to the best option from the available list.

CATEGORIES TO MATCH:
{main_text}

AVAILABLE OPTIONS:
{comp_text}

You must return valid JSON with this exact format:
{{"matches": [{{"CategorySlug": "value1", "MapMainSlug": "value2", "best_match": "exact_option_from_available_list"}}]}}

Rules:
- best_match must be exactly one of the available options
- Never use null or make up categories
- Always pick the closest available option"""

        try:
            response = call_ollama(prompt)
            
            if not response:
                print(f"  ✗ Empty response from AI")
                continue
            
            print(f"  Raw response length: {len(response)}")
            print(f"  First 100 chars: {response[:100]}")
            
            # Clean response more thoroughly
            response = response.strip()
            
            # Remove code blocks
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            
            response = response.strip()
            
            # Try to find JSON in the response
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start == -1 or end <= start:
                print(f"  ✗ No JSON found in response")
                print(f"  Full response: {response}")
                continue
            
            json_str = response[start:end]
            print(f"  Extracted JSON: {json_str[:200]}...")
            
            parsed = json.loads(json_str)
            
            if 'matches' in parsed:
                batch_matches = parsed['matches']
                # Validate matches are from the available list
                for match in batch_matches:
                    if match.get('best_match') not in comparison_categories:
                        print(f"Warning: '{match.get('best_match')}' not in available categories")
                        # Find closest match
                        best_match_lower = str(match.get('best_match', '')).lower()
                        for comp_cat in comparison_categories:
                            if comp_cat.lower() in best_match_lower or best_match_lower in comp_cat.lower():
                                match['best_match'] = comp_cat
                                break
                matches.extend(batch_matches)
                print(f"  ✓ Processed {len(batch_matches)} matches")
            else:
                print(f"  ✗ No 'matches' key found in response")
                
        except json.JSONDecodeError as e:
            print(f"  ✗ JSON error: {e}")
            print(f"  Problem with: {response[:300]}...")
        except Exception as e:
            print(f"  ✗ Batch failed: {e}")
    
    # Save results
    output = {"category_matches": matches}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Done! {len(matches)} matches saved to {output_file}")

# Usage
if __name__ == "__main__":
    # Change these filenames to yours
    main_file = "slugforai.json"  # ← PUT YOUR MAIN FILE HERE
    comparison_file = "categories_list.json"  # ← PUT YOUR COMPARISON FILE HERE  
    output_file = "matched_categories.json"  # ← PUT YOUR OUTPUT FILE HERE
    
    match_categories(main_file, comparison_file, output_file)