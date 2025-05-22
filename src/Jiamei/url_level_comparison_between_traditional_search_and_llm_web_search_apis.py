"""
Helper functions and Data Preparation for SEO 2.0 Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from serpapi import GoogleSearch
from urllib.parse import urlparse, urlunparse
from datetime import datetime
import pytz
import csv
import ast

# ---- Configuration ----
local_tz = pytz.timezone("America/Toronto")
SERPAPI_API_KEY = "YOUR_SERPAPI_API_KEY"  # Replace with your actual SerpAPI key

# ---- Helper Functions ----

def get_serpapi_urls(query, engine, num_results=10):
    params = {
        "engine": engine,
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": num_results
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results.get("organic_results", [])
    return [res["link"] for res in organic_results if res.get("link")]

def save_multiple_queries_to_csv(queries_by_category, engine="google",
                                 num_results=10, output_filename="serpapi_sources.csv"):
    timestamp = datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S")
    with open(output_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Query", "Category", "Source", "Source Rank",
                         "Timestamp", "Search Source"])
        for category, query in queries_by_category.items():
            urls = get_serpapi_urls(query, engine, num_results)
            for rank, url in enumerate(urls, start=1):
                writer.writerow([query, category, url, rank,
                                 timestamp, engine.title() + " Search"])
    print(f"✅ All results saved to {output_filename}")

def normalize_url(url):
    try:
        parsed = urlparse(url)
        cleaned = urlunparse((parsed.scheme, parsed.netloc,
                              parsed.path.rstrip("/"), '', '', ''))
        return cleaned.lower()
    except:
        return str(url).strip().lower()

def compare_2_urls_by_category(df1, df2, url_col1, url_col2, plot=False,
                                venn_labels=('File 1', 'File 2'),
                                summary_labels=('URLs in File1', 'URLs in File2')):
    def extract_urls(cell):
        try:
            if isinstance(cell, str) and cell.startswith("["):
                lst = ast.literal_eval(cell)
            else:
                lst = [cell]
            return [normalize_url(u) for u in lst if isinstance(u, str)]
        except:
            return []

    df1 = df1.copy()
    df2 = df2.copy()
    df1["Category"] = df1["Category"].str.lower()
    df2["Category"] = df2["Category"].str.lower()

    df1["Normalized URL"] = df1[url_col1].apply(extract_urls)
    df1 = df1.explode("Normalized URL")
    df2["Normalized URL"] = df2[url_col2].apply(extract_urls)
    df2 = df2.explode("Normalized URL")

    all_categories = sorted(set(df1["Category"]) | set(df2["Category"]))
    summary = []

    for category in all_categories:
        subset1 = df1[df1["Category"] == category]
        subset2 = df2[df2["Category"] == category]
        set1 = set(subset1["Normalized URL"].dropna())
        set2 = set(subset2["Normalized URL"].dropna())
        common = set1 & set2
        only1 = set1 - set2
        only2 = set2 - set1
        total = len(set1 | set2)
        overlap = len(common) / total * 100 if total else 0
        summary.append({
            "Category": category,
            summary_labels[0]: len(set1),
            summary_labels[1]: len(set2),
            "Common URLs": len(common),
            "Overlap Rate (%)": round(overlap, 2)
        })
        print(f"\n📦 Category: {category}")
        print(f"✅ Common ({len(common)}): {sorted(common)}")
        print(f"🔴 Only in {venn_labels[0]} ({len(only1)}): {sorted(only1)}")
        print(f"🔵 Only in {venn_labels[1]} ({len(only2)}): {sorted(only2)}")
        if plot:
            venn2([set1, set2], set_labels=venn_labels)
            plt.title(f"URL Overlap for {category}")
            plt.show()

    return pd.DataFrame(summary)

# ---- Data Preparation & Comparison ----

if __name__ == "__main__":
    # 1. Collect SERPAPI results
    queries = {"Instant Coffee": "Top recommended Instant Coffee in Canada"}
    save_multiple_queries_to_csv(queries, engine="google",
                                 num_results=20, output_filename="serpapi_google_5_5.csv")
    save_multiple_queries_to_csv(queries, engine="bing",
                                 num_results=20, output_filename="serpapi_bing_5_5.csv")

    # 2. Load collected data
    serpapi_google = pd.read_csv("serpapi_google_5_5.csv")
    serpapi_bing = pd.read_csv("serpapi_bing_5_5.csv")
    serpapi_google['Category'] = serpapi_bing['Category'] = "instant coffee"

    # 3. Load OpenAI processed data
    llm_raw = pd.read_csv("OpenAI_web_search_4_21_processed.csv")
    OpenAI_data = llm_raw[['brand', 'short_title', 'urls', 'category',
                           'URLs count', 'Mention Count',
                           'Avg Rank (Skip NaN)', 'Avg Rank (Fill 0)',
                           'Avg Rank (Fill 11)']].rename(columns={"category": "Category"})

    # 4. Compare Google vs Bing
    df_google_bing = compare_2_urls_by_category(
        serpapi_google, serpapi_bing,
        url_col1="Source", url_col2="Source",
        plot=True,
        venn_labels=('Google Search', 'Bing Search'),
        summary_labels=('URLs in Google Search', 'URLs in Bing Search')
    )
    print("\nGoogle vs Bing Summary:\n", df_google_bing)

    # 5. Compare OpenAI vs Google & Bing
    df_openai_google = compare_2_urls_by_category(
        OpenAI_data, serpapi_google,
        url_col1="urls", url_col2="Source",
        plot=True,
        venn_labels=('OpenAI Web Search', 'Google Search'),
        summary_labels=('URLs in OpenAI Web Search', 'URLs in Google Search')
    )
    print("\nOpenAI vs Google Summary:\n", df_openai_google)

    df_openai_bing = compare_2_urls_by_category(
        OpenAI_data, serpapi_bing,
        url_col1="urls", url_col2="Source",
        plot=True,
        venn_labels=('OpenAI Web Search', 'Bing Search'),
        summary_labels=('URLs in OpenAI Web Search', 'URLs in Bing Search')
    )
    print("\nOpenAI vs Bing Summary:\n", df_openai_bing)

