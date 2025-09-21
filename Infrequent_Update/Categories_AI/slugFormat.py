import json
import pandas as pd
import re

file_path_products = r'products.json'
file_path_slugs = r"slugs.json"

df_products = pd.read_json(file_path_products)
df_slugs = pd.read_json(file_path_slugs)

# Extract slugNr and convert to int
df_slugs['slugNr'] = df_slugs['category_slug'].str.extract(r'-(\d+)$').astype(int)

# Sort for ffill logic
df_slugsProcess = df_slugs.sort_values(by=["store_slug","slugNr"])

# Create MapMainSlug based on 'has_items' and forward fill
df_slugsProcess.loc[df_slugsProcess['is_main'] == True, 'MapMainSlug'] = df_slugsProcess['category_slug']
df_slugsProcess['MapMainSlug'] = df_slugsProcess['MapMainSlug'].ffill()

# Merge df_products with df_slugsProcess to add MapMainSlug
# Use left_on for df_products' column and right_on for df_slugsProcess' column
df_result = pd.merge(df_products, df_slugsProcess[['category_slug', 'MapMainSlug']],
                     left_on='CategorySlug',
                     right_on='category_slug', # This is from df_slugsProcess
                     how='left')

df_result['MapMainSlug'] = df_result['MapMainSlug'].fillna('')
# --- CRUCIAL CORRECTION HERE ---
# Modify the 'category_slug' column in df_result itself (which came from df_products on merge)
# And modify the 'MapMainSlug' column that was just merged into df_result
df_result['CategorySlug'] = df_result['CategorySlug'].str.replace(r'-\d+$', '', regex=True)
df_result['MapMainSlug'] = df_result['MapMainSlug'].str.replace(r'-\d+$', '', regex=True)


# Summarize by the modified category_slug and MapMainSlug
# Note: Ensure 'category_slug' from df_slugsProcess after merge does not conflict or you drop it if not needed.
# For simplicity and correctness, let's use the 'CategorySlug' from df_products for the groupby
# as it's the primary identifier you likely want to summarize.
summary_df = df_result.groupby(['CategorySlug', 'MapMainSlug']).size().reset_index(name='count')


path2=r'slugforai.json'

summary_df.to_json(path2, orient='records', indent=4)