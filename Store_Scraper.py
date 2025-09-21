import requests
import json
import time
import re
from datetime import datetime
import os
import subprocess
from Suport_Scraper.ai.FT_LORA_Ai_Units import initialize_model, process_rerun_items_inplace
import threading

# Venue URLs
venue_urls = [
   "https://consumer-api.wolt.com/consumer-api/consumer-assortment/v1/venues/slug/freshful-now-67ecf9a6e78872a14652406a/assortment/categories/slug/{}?language=ro",
    "https://consumer-api.wolt.com/consumer-api/consumer-assortment/v1/venues/slug/profi-baia-de-arama-3491-67fce8707ec55f4e5199f8d2/assortment/categories/slug/{}?language=ro",
    "https://consumer-api.wolt.com/consumer-api/consumer-assortment/v1/venues/slug/penny-4469-67ee32d9a0c535a55340303e/assortment/categories/slug/{}?language=ro",
   "https://consumer-api.wolt.com/consumer-api/consumer-assortment/v1/venues/slug/auchan-hypermarket-titan-67e2bd731248946a75c7a535/assortment/categories/slug/{}?language=ro",
   "https://consumer-api.wolt.com/consumer-api/consumer-assortment/v1/venues/slug/carrefour-hypermarket-mega-mall-9139-67ee8dde26be843d2832a717/assortment/categories/slug/{}?language=ro",
    "https://consumer-api.wolt.com/consumer-api/consumer-assortment/v1/venues/slug/kaufland-pantelimon-2470-67ecfaaae78872a1465240a6/assortment/categories/slug/{}?language=ro"

]

# Unit conversion factors
CONVERSION_FACTORS = {
    "kg": ("g", 1000),
    "l": ("ml", 1000),
    "spalari": ("buc", 1),
    "bucati": ("buc", 1),
    "buc": ("buc", 1),
    "gr": ("g", 1),
    "ml": ("ml", 1),
    "g": ("g", 1),
    "m": ("cm", 100),
    "cm": ("cm", 1),
    "cl": ("ml", 10),
    "v": ("v", 1),
    "pcs": ("buc", 1),
    "piese": ("buc", 1),
    "plicuri": ("buc", 1),
    "set": ("buc", 1),
    "cutii": ("buc", 1),
    "role": ("buc", 1),
    "capsule": ("buc", 1),
    "doze": ("buc", 1),
    "pahare": ("buc", 1),
    "pachete": ("buc", 1),
    "saci": ("buc", 1),
    "file":("buc", 1),
    "legatura":("buc", 1),
}

class FailedRequest:
    """Class to store information about failed requests for retry"""
    def __init__(self, url, request_type="main", base_url=None, category_id=None):
        self.url = url
        self.request_type = request_type  # "main" or "pagination"
        self.base_url = base_url
        self.category_id = category_id
        self.attempts = 0
        self.last_error = None

def evaluate_quantity(quant):
    """Safely evaluate a quantity string (e.g., "4*2" -> 8)"""
    try:
        quant = quant.replace("x", "*").replace("X", "*")
        return eval(quant)
    except (SyntaxError, NameError, TypeError):
        try:
            return float(quant)
        except ValueError:
            return None

def convert_to_smallest_unit(quant, unit):
    """Convert a quantity to the smallest unit"""
    if unit in CONVERSION_FACTORS:
        smallest_unit, factor = CONVERSION_FACTORS[unit]
        return quant * factor, smallest_unit
    return quant, unit

def load_extracted_quantities(processed_file_path):
    """Load extracted quantity/unit data and create title lookup dictionary"""
    title_lookup = {}
    
    if not processed_file_path or not os.path.exists(processed_file_path):
        print("⚠️ No processed quantities file found. Using original extraction method.")
        return title_lookup
    
    try:
        with open(processed_file_path, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
    
        valid_count = 0
        
        for item in processed_data:
            # Get title from the item
            title = item.get('title', '').strip()
            
            # Try to get quantity/unit from the top level first (your desired format)
            quantity = item.get('quantity')
            unit = item.get('unit')
            
            # If not found, try from original_item (fallback)
            if quantity is None or unit is None:
                original_item = item.get('original_item', {})
                quantity = original_item.get('Quantity', quantity)
                unit = original_item.get('Unit', unit)
            
            # If still not found, try from extracted_data JSON string
            if quantity is None or unit is None:
                extracted_data_str = item.get('extracted_data', '')
                if extracted_data_str:
                    try:
                        extracted_json = json.loads(extracted_data_str)
                        quantity = quantity or extracted_json.get('quantity')
                        unit = unit or extracted_json.get('unit')
                    except json.JSONDecodeError:
                        continue
            
            # Only add if we have both quantity and unit
            if title and quantity is not None and unit is not None and title not in title_lookup:
                title_lookup[title] = {
                    'Quantity': quantity,  # Capital Q to match your products format
                    'Unit': unit           # Capital U to match your products format
                }
                valid_count += 1
        
        print(f"✅ Loaded {valid_count} extracted quantities from processed file")
        return title_lookup
        
    except Exception as e:
        print(f"⚠️ Error loading processed file: {e}. Using original extraction method.")
        return {}

def preprocess_title(title):
    # Normalize spaces and fix common patterns

    # 6X0 33L → 6X0,33L or 6*0 33L
    title = re.sub(r'([x*])0\s+(\d+)(?=\s*(kg|g|gr|l|ml)\b)', r'\g<1>0,\2', title, flags=re.IGNORECASE)

    # 0 33L or 0 ,33L → 0.33L
    title = re.sub(r'\b0\s*[.,\s]\s*(\d+)(?=\s*(kg|g|gr|l|ml)\b)', r'0.\1', title, flags=re.IGNORECASE)

    # 033L → 0,33L
    title = re.sub(r'\b0(\d{2,})(?=\s*(kg|g|gr|l|ml)\b)', r'0,\1', title, flags=re.IGNORECASE)

    # FIXED: Handle /QUANTITYUNIT (e.g., /100G) by removing the slash and adding space
    title = re.sub(r'/\s*(\d+(?:[.,]\d+)?)\s*(kg|g|gr|l|ml)\b', r' \1\2', title, flags=re.IGNORECASE)

    return title

def extract_all_units_and_quantities(title):

    title_lower = title.lower()

    # Apply preprocessing first
    title_cleaned = preprocess_title(title_lower)   

    # Handle special cases first
    if "per bucata" in title_lower or "pe bucata" in title_lower:
        return 1, "buc", False
    
    if re.search(r"vrac\s*/?\s*100G", title_cleaned):
        return 100, "g", False
    
    if re.search(r"vrac\s*/?\s*kg", title_cleaned):
        return 1000, "g", False
    
    if re.search(r"\b(per|la)\s*(kg|bucata|buc|legatura)\b", title_cleaned):
        match = re.search(r"\b(kg|bucata|buc|legatura)\b", title_cleaned)
        if match:
            unit = "kg" if match.group(1) == "kg" else "buc"
            quant, smallest_unit = convert_to_smallest_unit(1, unit)
            return quant, smallest_unit, False

    # Main pattern matching
    indicators = r"spalari|bucati|plicuri|pachete|capsule|cutii|pahare|piese|saci|doze|role|file|buc|pcs|gr|kg|ml|set|cl|l|g"
    # Updated pattern to better handle the cleaned title
    pattern = rf"(\d+(?:[.,]\d+)?(?:\s*[x*]\s*\d+(?:[.,]\d+)?)?)\s*({indicators})\b"

    matches = re.findall(pattern, title_cleaned)
    
    if matches:
        last_match = matches[-1]
        quant = last_match[0].strip().replace(" ", "").replace(",", ".")
        unit = last_match[1].strip().lower()
        quant = evaluate_quantity(quant)

        if quant is not None:
            quant, unit = convert_to_smallest_unit(quant, unit)
            return quant, unit, False

    # Only fall back to vrac special case if no units were found at all
    if "vrac" in title_cleaned and not re.search(rf'\b({indicators})\b', title_cleaned):
        print(f"DEBUG: Vrac fallback triggered")  # Debug line
        return 1000, "g", False

    return 1, "buc", True

def get_with_retries(url, max_retries=10, initial_wait=1, max_wait=20):
    """Make HTTP request with exponential backoff retry logic"""
    wait_time = initial_wait
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            
            # Success cases
            if response.status_code == 200:
                return response
            elif response.status_code == 404:
                return response  # Let caller handle 404s
            elif response.status_code == 429:
                print(f"429 Too Many Requests. Waiting {wait_time} seconds before retrying (Attempt {attempt + 1})...")
            else:
                print(f"HTTP {response.status_code}. Waiting {wait_time} seconds before retrying (Attempt {attempt + 1})...")
            
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, max_wait)  # Exponential backoff with cap
            
        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt + 1}. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            wait_time = min(wait_time * 1.5, max_wait)
        except requests.exceptions.ConnectionError:
            print(f"Connection error on attempt {attempt + 1}. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            wait_time = min(wait_time * 1.5, max_wait)
        except Exception as e:
            print(f"Request error on attempt {attempt + 1}: {e}")
            time.sleep(wait_time)
            wait_time = min(wait_time * 1.5, max_wait)
    
    raise Exception(f"Failed after {max_retries} retries")

def get_with_infinite_retries(url, initial_wait=1, max_wait=21):
    """Make HTTP request with infinite retries until success"""
    wait_time = initial_wait
    attempt = 0
    
    while True:
        attempt += 1
        try:
            response = requests.get(url, timeout=30)
            
            # Success cases
            if response.status_code == 200:
                return response
            elif response.status_code == 404:
                return response  # Let caller handle 404s
            elif response.status_code == 429:
                print(f"429 Too Many Requests. Waiting {wait_time} seconds before retrying (Attempt {attempt})...")
            else:
                print(f"HTTP {response.status_code}. Waiting {wait_time} seconds before retrying (Attempt {attempt})...")
            
            time.sleep(wait_time)
            wait_time = min(wait_time * 1.2, max_wait)  # Gradual backoff with cap
            
        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt}. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            wait_time = min(wait_time * 1.1, max_wait)
        except requests.exceptions.ConnectionError:
            print(f"Connection error on attempt {attempt}. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            wait_time = min(wait_time * 1.1, max_wait)
        except Exception as e:
            print(f"Request error on attempt {attempt}: {e}. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            wait_time = min(wait_time * 1.1, max_wait)

def chateg():
    with open(r'Suport_Scraper/matched_categories.json', 'r', encoding='utf-8') as f1:
        reader1 = json.load(f1)
    # Extract the category_matches array from the JSON structure
    fdata = reader1.get('category_matches', [])
    return fdata

def data_getting(url, data, all_data, slugs_data, store_main_slugs, title_lookup):
    """
    Process API response and extract product data, tracking main slugs per store.
    Now includes real-time quantity/unit updates from processed data.
    Returns True if this was a main slug, False otherwise.
    """
    # Extract store name
    after_slug = url.split('/slug/')[1]
    before_assortment = after_slug.split('/assortment')[0]
    parts = re.split(r'\d', before_assortment, maxsplit=1)
    store_name = parts[0].rstrip('-')

    categtoreplace = chateg()
     
    def clean(category_string):
        return re.sub(r'[^a-z]', '', category_string.lower())

    # Create mapping for category+main_slug combination to best_match
    combined_category_map = {}
    for row in categtoreplace:
        # Check if row is a dictionary (handle malformed JSON)
        if not isinstance(row, dict):
            print(f"Warning: Skipping malformed row in matched_categories.json: {row}")
            continue
            
        category_slug = row.get('CategorySlug', '')
        map_main_slug = row.get('MapMainSlug', '')
        best_match = row.get('best_match', 'Miscellaneous')
        
        # Only create mapping if we have valid data
        if category_slug and map_main_slug:
            # Create cleaned combined key
            combined_key = clean(category_slug + map_main_slug)
            combined_category_map[combined_key] = best_match
        else:
            print(f"Warning: Missing CategorySlug or MapMainSlug in row: {row}")

    # Get items (handle as list)
    items = data.get('items', [])
    if not isinstance(items, list):
        items = [items] if items else []

    # Get category information
    category_info = data.get('category', {})
    category_name = category_info.get('name', 'N/A')
    category_slug = category_info.get('slug', 'N/A')

    category_api_data = data.get('category', {})
    img = category_api_data.get('images', [])
    if not isinstance(img, list):
        img = [img] if img else []

    is_main = False # Initialize with a boolean False

    if not items and bool(img): 
        is_main = True
        # Update the main slug for this store
        store_main_slugs[store_name] = category_slug
        print(f"🔄 Updated main slug for {store_name}: {category_slug}")
    
    # Save slug and store information
    slug_entry = {
        "category_name": category_name,
        "category_slug": category_slug,
        "store_name": store_name,
        "store_slug": before_assortment,
        "has_items": bool(items),
        "is_main": is_main
    }
    
    # Add to slugs data if not already present
    existing_slug = next((s for s in slugs_data if s["category_slug"] == category_slug and s["store_slug"] == before_assortment), None)
    if not existing_slug:
        slugs_data.append(slug_entry)

    # Process items (only if there are any)
    for item in items:
        # Safe price conversion
        current_priceunrd = item.get('price', 0) / 100 if item.get('price') else 0
        current_price = round(current_priceunrd, 2)
        old_price = item.get('original_price', 0) / 100 if item.get('original_price') else 0

        # Get image URL
        image_url = "N/A"
        if item.get('images'):
            if isinstance(item['images'], list) and item['images']:
                image_url = item['images'][0].get('url', "N/A")
            elif isinstance(item['images'], dict):
                image_url = item['images'].get('url', "N/A")

        # Build product data
        prod_id = item.get('id', "N/A")
        product_link = f"https://wolt.com/en/rou/bucharest/venue/{before_assortment}/{prod_id}"
        title = item.get('name', "N/A")

        # *** ENHANCED: Check for extracted quantities first ***
        if title.strip() in title_lookup:
            extracted_data = title_lookup[title.strip()]
            quant = extracted_data['Quantity']
            unit = extracted_data['Unit']
            reRun_flag = False
            # Convert to smallest unit for consistency
            quant, unit = convert_to_smallest_unit(quant, unit)
            reRun_flag = False  # Mark as processed by extraction
            print(f"✅ Using extracted data for: {title[:50]}... (Q:{quant}, U:{unit})")
        else:
            # Fallback to original extraction method
            quant, unit, reRun_flag = extract_all_units_and_quantities(title)

        # Calculate price per unit metric
        try:
            current_price = float(current_price)
            quant = float(quant)
            price_metric = current_price / quant if quant and current_price else 0
        except (ValueError, TypeError, ZeroDivisionError):
            price_metric = 0
            print(f"Price calculation error - Title: {title}, Quantity: {quant}, Price: {current_price}")

        if price_metric < 0.0003:
            low_valflag = "Low"
        else:
            low_valflag = "Norm"

        cleanedcategory_name = clean(category_name)

        # Get the current main slug for this store
        current_main_slug = store_main_slugs.get(store_name, "N/A")
        
        # Create combined key for matching (catheg + main_slug cleaned)
        combined_key = clean(category_slug + current_main_slug)
        
        # Get best match using combined key, fallback to "Miscellaneous"
        useChateg = combined_category_map.get(combined_key, "Miscellaneous")

        category_data = {
            "Image URL": image_url,
            "Current Price": current_price,
            "Old Price": old_price,
            "Description": item.get('description', "N/A"),
            "Title": title,
            "Store": store_name,
            "Product Link": product_link,
            "Prod ID": prod_id,
            "Unit": unit,
            "MetrPrice": price_metric,
            "Quantity": quant,
            "LowValFlag": low_valflag,
            "categoryname": category_name,
            "usecategory": useChateg,
            "CategorySlug": category_slug,
            "main_slug": current_main_slug,
            "Rerun_Flag": reRun_flag  # This will be False for extracted data
        }

        all_data.append(category_data)
    
    return is_main

def process_category(base_url, category_id, all_data, slugs_data, failed_requests, store_main_slugs, title_lookup):
    """Process a single category and handle pagination"""
    url = base_url.format(category_id)
    
    try:
        response = get_with_retries(url)
        
        if response.status_code == 404:
            return False  # No more categories
        
        data = response.json()
        
        # Check if category not found
        if 'detail' in data and 'not found' in data['detail']:
            return False  # No more categories
        
        category_name = data.get('category', {}).get('name', 'N/A')
        print(f"Category {category_id}: {category_name}")
        
        # Process main page
        is_main = data_getting(url, data, all_data, slugs_data, store_main_slugs, title_lookup)
        
        # Handle pagination
        nextpt = data.get('metadata', {}).get('next_page_token')
        while nextpt:
            urlpg = url + "&page_token=" + nextpt
            try:
                response = get_with_retries(urlpg)
                data = response.json()
                data_getting(url, data, all_data, slugs_data, store_main_slugs, title_lookup)
                nextpt = data.get('metadata', {}).get('next_page_token')
            except Exception as e:
                print(f"Failed pagination request for category {category_id}: {e}")
                failed_req = FailedRequest(urlpg, "pagination", base_url, category_id)
                failed_req.last_error = str(e)
                failed_requests.append(failed_req)
                break  # Stop pagination for this category, will retry later
                
        return True  # Successfully processed category
        
    except Exception as e:
        print(f"Failed main request for category {category_id}: {e}")
        failed_req = FailedRequest(url, "main", base_url, category_id)
        failed_req.last_error = str(e)
        failed_requests.append(failed_req)
        return True  # Continue to next category

def retry_failed_requests(failed_requests, all_data, slugs_data, store_main_slugs, title_lookup):
    """Retry all failed requests until they succeed"""
    if not failed_requests:
        return
        
    print(f"\n=== RETRYING {len(failed_requests)} FAILED REQUESTS UNTIL SUCCESS ===")
    
    while failed_requests:
        print(f"\n{len(failed_requests)} requests remaining to retry...")
        
        # Create a copy of failed requests to iterate over
        current_failures = failed_requests.copy()
        failed_requests.clear()  # Clear the original list
        
        for failed_req in current_failures:
            failed_req.attempts += 1
            print(f"Retrying (attempt {failed_req.attempts}): Category {failed_req.category_id} - {failed_req.request_type}")
            
            try:
                if failed_req.request_type == "main":
                    # Retry main category request with infinite retries
                    response = get_with_infinite_retries(failed_req.url)
                    
                    if response.status_code == 404:
                        print(f"✓ Category {failed_req.category_id} not found (404 - expected)")
                        continue  # Don't add back to failed requests
                    
                    data = response.json()
                    if 'detail' in data and 'not found' in data['detail']:
                        print(f"✓ Category {failed_req.category_id} not found (expected)")
                        continue  # Don't add back to failed requests
                    
                    # Successfully got data, process it
                    data_getting(failed_req.url, data, all_data, slugs_data, store_main_slugs, title_lookup)
                    
                    # Also handle pagination for retried main requests
                    nextpt = data.get('metadata', {}).get('next_page_token')
                    while nextpt:
                        urlpg = failed_req.url + "&page_token=" + nextpt
                        try:
                            response = get_with_infinite_retries(urlpg)
                            data = response.json()
                            data_getting(failed_req.url, data, all_data, slugs_data, store_main_slugs, title_lookup)
                            nextpt = data.get('metadata', {}).get('next_page_token')
                        except Exception as e:
                            print(f"Failed pagination in retry: {e}")
                            new_failed = FailedRequest(urlpg, "pagination", failed_req.base_url, failed_req.category_id)
                            new_failed.last_error = str(e)
                            failed_requests.append(new_failed)
                            break
                    
                    print(f"✓ Successfully retried main request for category {failed_req.category_id}")
                        
                elif failed_req.request_type == "pagination":
                    # Retry pagination request with infinite retries
                    response = get_with_infinite_retries(failed_req.url)
                    data = response.json()
                    data_getting(failed_req.base_url.format(failed_req.category_id), data, all_data, slugs_data, store_main_slugs, title_lookup)
                    print(f"✓ Successfully retried pagination request for category {failed_req.category_id}")
                    
            except KeyboardInterrupt:
                print("\n🛑 Retry process interrupted by user")
                failed_requests.extend(current_failures[current_failures.index(failed_req):])
                break
            except Exception as e:
                error_msg = str(e)
                print(f"✗ Unexpected error for category {failed_req.category_id}: {error_msg}")
                failed_req.last_error = error_msg
                failed_requests.append(failed_req)  # Add back to retry queue
            
            # Add a small delay between individual retries
            time.sleep(0.2)
        
        # If we have requests that still need retrying, wait before next batch
        if failed_requests:
            wait_time = min(2 + len(current_failures) * 0.1, 10)  # Short wait between batches
            print(f"Waiting {wait_time:.1f} seconds before next retry batch...")
            time.sleep(wait_time)
    
    print(f"\n✅ All failed requests successfully retried!")

def remove_duplicates(products):
    """Remove duplicate products based on image URL"""
    unique_products = {}
    for product in reversed(products):
        image_url = product.get("Image URL")
        if image_url not in unique_products:
            unique_products[image_url] = product
    return list(reversed(unique_products.values()))

def main():
    """Main scraping function"""
    # Start model initialization in background thread
    model_thread = threading.Thread(target=initialize_model)
    model_thread.start()
    print("🚀 Model initialization started in background...")
    
    # Continue with the rest of your code immediately
    title_lookup = load_extracted_quantities(r"Suport_Scraper/AI_Proccesed_Units.json")
    active_venues = {url: True for url in venue_urls}
    all_data = []
    slugs_data = []
    failed_requests = []
    store_main_slugs = {}
 
    # Your scraping logic continues here...
    i = 1
    while any(active_venues.values()):
        for base_url in venue_urls:
            if not active_venues[base_url]:
                continue

            success = process_category(base_url, i, all_data, slugs_data, failed_requests, store_main_slugs, title_lookup)
            if not success:
                print(f"Done with venue at slug {i}")
                active_venues[base_url] = False

            time.sleep(0.1)

        i += 1

    # Wait for model initialization to complete before post-processing
    print("⏳ Waiting for model initialization to complete...")
    model_thread.join()  # This will wait until initialize_model() finishes
    print("✅ Model initialization completed!")

    # Continue with rest of your code...
    retry_failed_requests(failed_requests, all_data, slugs_data, store_main_slugs, title_lookup) 
    # Remove duplicates
    unique_products = remove_duplicates(all_data)
    
    print(f"\nScraping complete! Collected {len(unique_products)} unique products")
    print(f"Collected {len(slugs_data)} unique category slugs")
    print(f"Main slugs tracked for stores: {store_main_slugs}")
    if failed_requests:
        print(f"⚠️  Warning: {len(failed_requests)} requests could not be recovered")

    today = datetime.now()
    date_str = today.strftime("%d-%m-%Y")

    with open(rf'Scraped_Data/products/products_{date_str}.json', 'w', encoding='utf-8') as json_file:
       json.dump(unique_products, json_file, indent=4, ensure_ascii=False)

    print(f"Product data saved to products_{date_str}.json")
    
    # Save slugs data
    with open(r'Scraped_Data/slugs.json', 'w', encoding='utf-8') as json_file:
       json.dump(slugs_data, json_file, indent=4, ensure_ascii=False)
    
    print("Slugs data saved to slugs.json")
    subprocess.Popen(["python", r"Suport_Scraper/ai/Vectorize_Data.py"])
    
    # Process with AI and get results
    results = process_rerun_items_inplace(rf'Scraped_Data/products/products_{date_str}.json')
    print(f"✅ Unsloth post-processing finished: {results}")
    
    # Recalculate MetrPrice and LowValFlag after AI processing
    print("🔄 Recalculating MetrPrice and LowValFlag...")
    
    # Load the processed data (results might be a modified list or the file might be updated)
    # Let's load from file to be safe
    with open(rf'Scraped_Data/products/products_{date_str}.json', 'r', encoding='utf-8') as f:
        products = json.load(f)
    
    for product in products:
        quantity = product.get("Quantity", 1)
        
        # If previously flagged as Low, set quantity to 1 buc
        if product.get("LowValFlag") == "Low":
            quantity = 1
            product["Quantity"] = quantity
            product["Unit"] = "buc"  # Also update unit to buc

        # Recalculate MetrPrice
        price = product.get("Current Price", 0)
        try:
            product["MetrPrice"] = round(price / quantity, 6) if quantity else 0
        except Exception as e:
            print(f"Error recalculating MetrPrice for {product.get('Title', '')}: {e}")
            product["MetrPrice"] = 0

        # Recalculate LowValFlag
        product["LowValFlag"] = "Low" if product["MetrPrice"] < 0.0003 else "Norm"

    # Save updated results back to JSON
    with open(rf'Scraped_Data/products/products_{date_str}.json', 'w', encoding='utf-8') as f:
        json.dump(products, f, indent=4, ensure_ascii=False)

    print(f"✅ Products file updated with recalculated MetrPrice and LowValFlag")

if __name__ == "__main__":
    main()