import pandas as pd
import re
import time
import pytz
import csv
import glob
import matplotlib.pyplot as plt
from urllib.parse import urlparse, urlunparse
from datetime import datetime
from matplotlib_venn import venn2

# ========== CONFIGURATION ==========
PERPLEXITY_API_KEY = "your perplexity api key"
local_tz = pytz.timezone("America/Toronto")

# Dummy client for placeholder
class OpenperplexSync:
    def __init__(self, key):
        self.key = key

    def search(self, query, model, response_language, answer_type):
        return {"llm_response": "Simulated response for testing."}

client_sync = OpenperplexSync(PERPLEXITY_API_KEY)

# ========== FUNCTION DEFINITIONS ==========

def extract_product_info(response_text):
    product_info = []
    product_blocks = response_text.strip().split('\n\n')
    for block in product_blocks:
        title = re.search(r"Product Title:\s*(.*)", block)
        description = re.search(r"Description:\s*(.*)", block)
        brand = re.search(r"Brand:\s*(.*)", block)
        retailer = re.search(r"Retailer:\s*(.*)", block)
        source = re.search(r"Source URL:\s*(http[s]?://\S+)", block)
        comments = re.search(r"Comments:\s*(.*)", block)
        product_info.append({
            "Title": title.group(1) if title else "N/A",
            "Description": description.group(1) if description else "N/A",
            "Brand": brand.group(1) if brand else "N/A",
            "Retailer": retailer.group(1) if retailer else "N/A",
            "Source": source.group(1) if source else "N/A",
            "Comments": comments.group(1) if comments else "N/A"
        })
    return product_info

def fetch_perplexity_products(query, csv_filename, product_num=10):
    data = []
    full_query = f"""
    Provide the top {product_num} {query}.
    For each product, return:
    Product Title: [Title]
    Description: [Description]
    Brand: [Brand]
    Retailer: [Retailer]
    Source URL: [URL]
    Comments: [Comments]
    """

    print(f"\nPerplexity Searching for query: {query}")
    try:
        response = client_sync.search(
            query=full_query,
            model='o3-mini-high',
            response_language="en",
            answer_type="text"
        )
        response_text = response.get("llm_response", "")
        product_info = extract_product_info(response_text)
        if product_info:
            timestamp = datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S")
            for info in product_info:
                data.append({
                    **info,
                    "Search Source": "Perplexity",
                    "Query": query,
                    "Timestamp": timestamp
                })
            time.sleep(2)
    except Exception as e:
        print(f"Error fetching Perplexity API for {query}: {e}")

    if data:
        df = pd.DataFrame(data)
        df.to_csv(csv_filename, index=False)
        print(f"✅ CSV saved: {csv_filename}")
    else:
        print("⚠️ No data collected.")
        return pd.DataFrame()

def normalize_url(url):
    try:
        parsed = urlparse(url)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), '', '', '')).lower()
    except:
        return str(url).lower().strip()

def compare_2_urls_by_category(googlefile, llmfile, plot=False):
    df1 = pd.read_csv(googlefile)
    df2 = pd.read_csv(llmfile)

    df1["Normalized URL"] = df1["Source"].apply(normalize_url)
    df2["Normalized URL"] = df2["Source"].apply(normalize_url)

    all_categories = sorted(set(df1["Category"]) | set(df2["Category"]))
    summary = []

    for category in all_categories:
        urls1 = df1[df1["Category"] == category]
        urls2 = df2[df2["Category"] == category]
        urls1_set = set(urls1["Normalized URL"].dropna())
        urls2_set = set(urls2["Normalized URL"].dropna())

        common = urls1_set & urls2_set
        only_in_1 = urls1_set - urls2_set
        only_in_2 = urls2_set - urls1_set

        total_unique = len(urls1_set | urls2_set)
        overlap_rate = len(common) / total_unique if total_unique > 0 else 0

        summary.append({
            "Category": category,
            "URLs in Google": len(urls1_set),
            "URLs in LLM": len(urls2_set),
            "Common URLs": len(common),
            "Top Google Rank of Common": urls1[urls1['Normalized URL'].isin(common)]['Source Rank'].min() if common else "N/A",
            "Overlap Rate": f"{round(overlap_rate*100, 2)}%"
        })

        print(f"\n📦 {category}: {len(common)} common URLs")
        if plot:
            venn2([urls1_set, urls2_set], set_labels=("Google", "LLM"))
            plt.title(f"URL Overlap: {category}")
            plt.show()

    return pd.DataFrame(summary)

def save_serpapi_urls_to_csv(query, num_results=10, output_filename="serpapi_sources.csv"):
    # Dummy implementation placeholder
    print(f"(Simulated) Saved URLs for query '{query}' to {output_filename}")

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    queries = [
        "best instant coffee",
        "best baking ingredients",
        "best cat food",
        "best dog food",
        "best cat litter"
    ]

    for q in queries:
        save_serpapi_urls_to_csv(q, num_results=10, output_filename=f"GoogleSearch_{q.replace(' ', '_')}.csv")
        # fetch_perplexity_products(q, csv_filename=f"perplex_{q.replace(' ', '_')}.csv", product_num=10)

    csv_files = glob.glob("GoogleSearch_*.csv")
    df_list = []

    for file in csv_files:
        df = pd.read_csv(file)
        category = file.replace("GoogleSearch_", "").replace(".csv", "").replace("_", " ")
        df["Category"] = category
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv("merged_GoogleSearch_Data.csv", index=False)
    print("✅ Merged Google Search data into 'merged_GoogleSearch_Data.csv'")

    # Comparison example (filenames must exist)
    # compare_2_urls_by_category('merged_GoogleSearch_Data.csv', 'merged_perplexity_Data.csv', plot=False)
    print("✅ Script completed.")