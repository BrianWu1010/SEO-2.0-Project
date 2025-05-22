# Required installations
# pip install openai==0.28
# pip install google-search-results pandas requests praw --upgrade openperplex

import requests
from bs4 import BeautifulSoup
import re
import time
import json
import csv
import openai
import praw
from serpapi import GoogleSearch
from datetime import datetime
import pytz
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
from openperplex import OpenperplexSync

# ====== API CONFIGURATION ======
openai.api_key = "your openai api key"
SERPAPI_API_KEY = "your serpapi key"
local_tz = pytz.timezone("America/Toronto")

headers = {"User-Agent": "Mozilla/5.0"}

# ====== HELPER FUNCTIONS ======

def get_serpapi_urls(query, engine='google', num_results=10):
    params = {
        "engine": engine,
        "q": query,
        "cc": "CA",
        "api_key": SERPAPI_API_KEY,
        "num": num_results
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results.get("organic_results", [])
    return [res.get("link") for res in organic_results if res.get("link")]

def get_cleaned_html(url):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["style"]):
            tag.decompose()
        article = soup.find("article") or soup.find("div", {"class": "content"}) or soup
        return article.get_text(separator="\n", strip=True)
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def extract_product_data(html):
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert at extracting product rankings and reviews from HTML."},
                {"role": "user", "content": html}
            ],
            functions=[{
                "name": "parse_product_data",
                "description": "Extract product ranking and review info.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "products": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "Result Rank": {"type": "integer"},
                                    "Product Title": {"type": "string"},
                                    "Comment": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            }],
            function_call={"name": "parse_product_data"}
        )
        func_call = completion.choices[0].message.get("function_call", {})
        data = json.loads(func_call.get("arguments", "{}"))
        return data.get("products", [])
    except Exception as e:
        print(f"Error extracting product data: {e}")
        return None

def get_reddit_comments(url):
    reddit = praw.Reddit(
        client_id="your client id",
        client_secret="your client secret",
        user_agent="YourScraper/1.0",
        check_for_async=False
    )
    submission = reddit.submission(url=url)
    submission.comments.replace_more(limit=0)
    comments = submission.comments.list()
    top_comments = sorted(comments, key=lambda x: x.score, reverse=True)[:10]
    return "\n\n".join([f"Score: {c.score}\nComment: {c.body}" for c in top_comments])

def extract_product_data_from_reddit(url):
    try:
        reddit_comments = get_reddit_comments(url)
        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Extract positively reviewed products from Reddit comments."},
                {"role": "user", "content": reddit_comments}
            ],
            functions=[{
                "name": "parse_product_data",
                "description": "Parse product info from Reddit.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "products": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "Result Rank": {"type": "integer"},
                                    "Product Title": {"type": "string"},
                                    "Comment": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            }],
            function_call={"name": "parse_product_data"}
        )
        func_call = completion.choices[0].message.get("function_call", {})
        data = json.loads(func_call.get("arguments", "{}"))
        return data.get("products", [])
    except Exception as e:
        print(f"Reddit extraction failed: {e}")
        return None

def scrape_serpapi_products(query, engine='google', required_count=5, num_results=10, csv_filename="output.csv"):
    urls = get_serpapi_urls(query, engine, num_results)
    print(f"Fetched {len(urls)} URLs from {engine} for '{query}'")

    results_set = set()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for i, url in enumerate(urls, start=1):
        if len(results_set) >= required_count:
            break

        print(f"\n🔍 Scraping URL {i}: {url}")
        if "reddit" in url.lower():
            products_data = extract_product_data_from_reddit(url)
        else:
            cleaned_html = get_cleaned_html(url)
            if not cleaned_html or len(cleaned_html) < 100:
                print(f"⚠️ Skipping {url} due to insufficient content.")
                continue
            products_data = extract_product_data(cleaned_html)

        if not products_data:
            continue

        for product in products_data:
            product["Source Link"] = url
            product["Source Rank"] = i
            product["Timestamp"] = timestamp

        hashable = tuple(frozenset(prod.items()) for prod in products_data)
        results_set.add(hashable)
        time.sleep(1)

    all_products = [dict(item) for group in results_set for item in group]

    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Result Rank", "Title", "Description", "Source", "Source Rank", "Search Source", "Query", "Timestamp"])
        for prod in all_products:
            writer.writerow([
                prod.get("Result Rank", "N/A"),
                prod.get("Product Title", "N/A"),
                prod.get("Comment", "N/A"),
                prod.get("Source Link", "N/A"),
                prod.get("Source Rank", "N/A"),
                engine,
                query,
                prod.get("Timestamp", timestamp)
            ])
    print(f"✅ Saved results to {csv_filename}")

# ====== EXECUTION BLOCK ======
if __name__ == "__main__":
    categories = [
        "Best baking ingredients",
        "Best instant coffee",
        "Best dog food",
        "Best cat food",
        "Best cat litter"
    ]

    for query in categories:
        filename = f"GoogleSearch_{query.lower().replace(' ', '_')}.csv"
        scrape_serpapi_products(query, required_count=5, num_results=10, csv_filename=filename)

    csv_files = glob.glob("GoogleSearch_*.csv")
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file)
        df["Result Rank"] = df.groupby("Source")["Result Rank"].transform(lambda x: x.fillna(pd.Series(range(1, len(x) + 1), index=x.index)))
        category = file.replace("GoogleSearch_", "").replace(".csv", "").replace("_", " ")
        df["Category"] = category
        df_list.append(df)

    merged = pd.concat(df_list, ignore_index=True)
    merged.to_csv("GoogleSearch_Data_4_15.csv", index=False)
    print("✅ Merged all into 'GoogleSearch_Data_4_15.csv'")