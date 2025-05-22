# Required imports
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import openai  # make sure you have `openai` installed

"""
Helper Functions
"""

def extract_brand_from_title(title):
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You are an expert at identifying real product brand names from product titles. "
                    "Only return the brand name if it is verifiably real. Do not invent or guess. "
                    "If you cannot determine a valid brand, return 'Unknown'."
                )},
                {"role": "user", "content": title}
            ]
        )
        brand = completion.choices[0].message["content"].strip()
        return brand
    except Exception as e:
        print(f"Error processing title: {title}. Error: {e}")
        return "Unknown"

def clean_name(name):
    return str(name).lower().strip().replace("®", "").replace("™", "").replace("-", "").replace("’", "'").replace("é", "e")

def standardize_brand(df, brand_column="Brand"):
    df.loc[df[brand_column].str.contains("hill's", case=False, na=False), brand_column] = "hill's science diet"
    df.loc[df[brand_column].str.contains("wellness", case=False, na=False), brand_column] = "wellness"
    df.loc[df[brand_column].str.contains("purina", case=False, na=False), brand_column] = "purina"
    df.loc[df[brand_column].str.contains("fancy feast", case=False, na=False), brand_column] = "purina"
    df.loc[df[brand_column].str.contains("blue bottle", case=False, na=False), brand_column] = "blue bottle coffee"
    df.loc[df[brand_column].str.contains("canyon", case=False, na=False), brand_column] = "canyon coffee"
    df.loc[df[brand_column].str.contains("kloo", case=False, na=False), brand_column] = "kloo coffee"
    df.loc[df[brand_column].str.contains("king arthur", case=False, na=False), brand_column] = "king arthur baking company"
    return df

"""
Preprocessing
"""

googlesearch_df = pd.read_csv("updated_googlesearching.csv")
llm_df = pd.read_csv("LLM_Websearch_Data_3_27.csv")
googleshop_df = pd.read_csv("updated_googleshopping.csv")

for df in [googlesearch_df, llm_df, googleshop_df]:
    df["Brand"] = df["Brand"].map(clean_name)
llm_df["Title"] = llm_df["Title"].map(clean_name)

googlesearch_df = standardize_brand(googlesearch_df)
llm_df = standardize_brand(llm_df)
googleshop_df = standardize_brand(googleshop_df)

"""
Comparison
"""

def compare_brands(category, googlesearch_df, llm_df, googleshop_df):
    print("=" * 60)
    print(f"\n📦 Category: {category.title()}")

    google_cat = googlesearch_df[googlesearch_df["Category"].str.lower() == category.lower()]
    google_brands = google_cat["Brand"].str.lower().replace("unknown", "unknown_google").dropna().unique()

    llm_cat = llm_df[llm_df["Product Title"].str.lower().str.contains(category.lower())]
    llm_brands = llm_cat["Brand"].str.lower().replace("unknown", "unknown_llm").dropna().unique()

    shop_cat = googleshop_df[googleshop_df["Product title"].str.lower() == category.lower()]
    shop_brands = shop_cat["Brand"].str.lower().replace("unknown", "unknown_shop").dropna().unique()

    google_set = set(google_brands)
    llm_set = set(llm_brands)
    shop_set = set(shop_brands)

    common_all = google_set & llm_set & shop_set
    common_google_llm = (google_set & llm_set) - shop_set
    common_google_shop = (google_set & shop_set) - llm_set
    common_llm_shop = (llm_set & shop_set) - google_set
    only_google = google_set - (llm_set | shop_set)
    only_llm = llm_set - (google_set | shop_set)
    only_shop = shop_set - (google_set | llm_set)

    if common_all: print("✅ Common brands (all three):", sorted(common_all))
    if common_google_llm: print("🔹 Google & LLM:", sorted(common_google_llm))
    if common_google_shop: print("🔹 Google & Shopping:", sorted(common_google_shop))
    if common_llm_shop: print("🔹 LLM & Shopping:", sorted(common_llm_shop))
    if only_google: print("🔸 Only in Google Search:", sorted(only_google))
    if only_llm: print("🔸 Only in LLM WebSearch:", sorted(only_llm))
    if only_shop: print("🔸 Only in Google Shopping:", sorted(only_shop))

    venn3([google_set, llm_set, shop_set], set_labels=("Google Search", "LLM WebSearch", "Google Shopping"))
    plt.title(f"Brand Overlap for {category.title()}")
    plt.show()

categories = ["dog food", "cat food", "cat litter", "instant coffee", "baking ingredients"]
for cat in categories:
    compare_brands(cat, googlesearch_df, llm_df, googleshop_df)
