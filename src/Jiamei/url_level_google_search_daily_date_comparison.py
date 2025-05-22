import pandas as pd
from urllib.parse import urlparse, urlunparse
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3

# --- Normalize URL ---
def normalize_url(url):
    try:
        parsed = urlparse(url)
        clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), '', '', ''))
        return clean_url.lower()
    except:
        return str(url).lower().strip()

# --- Compare 2-Day Overlap ---
def compare_2_urls_by_category(file1, file2, plot=False):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    df1["Normalized URL"] = df1["Source Link"].apply(normalize_url)
    df2["Normalized URL"] = df2["Source Link"].apply(normalize_url)

    all_categories = sorted(set(df1["Category"].unique()) | set(df2["Category"].unique()))
    summary = []

    for category in all_categories:
        urls1 = df1[df1["Category"] == category]
        urls2 = df2[df2["Category"] == category]

        urls1_set = set(urls1["Normalized URL"].dropna().unique())
        urls2_set = set(urls2["Normalized URL"].dropna().unique())

        common = urls1_set & urls2_set
        only_in_1 = urls1_set - urls2_set
        only_in_2 = urls2_set - urls1_set

        total_unique = len(urls1_set | urls2_set)
        overlap_rate = len(common) / total_unique if total_unique > 0 else 0

        summary.append({
            "Category": category,
            "URLs in Day1": len(urls1_set),
            "URLs in Day2": len(urls2_set),
            "Common URLs": len(common),
            "Overlap Rate": round(overlap_rate, 2)
        })

        print(f"\n📦 Category: {category}")
        print(f"✅ Common URLs: {len(common)}")

        print(f"\n🔴 Only in {file1} ({len(only_in_1)}):")
        print(urls1[urls1["Normalized URL"].isin(only_in_1)]
              [["Source Link", "Source Rank"]]
              .drop_duplicates(subset="Source Link")
              .sort_values("Source Rank"))

        print(f"\n🔵 Only in {file2} ({len(only_in_2)}):")
        print(urls2[urls2["Normalized URL"].isin(only_in_2)]
              [["Source Link", "Source Rank"]]
              .drop_duplicates(subset="Source Link")
              .sort_values("Source Rank"))

        if plot:
            venn2([urls1_set, urls2_set], set_labels=("Day 1", "Day 2"))
            plt.title(f"URL Overlap for {category}")
            plt.show()

    return pd.DataFrame(summary)

# --- Compare 3-Day Overlap ---
def compare_google_urls_by_category_3(file1, file2, file3, plot=False):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)

    df1["Normalized URL"] = df1["Source Link"].apply(normalize_url)
    df2["Normalized URL"] = df2["Source Link"].apply(normalize_url)
    df3["Normalized URL"] = df3["Source Link"].apply(normalize_url)

    all_categories = sorted(set(df1["Category"].unique()) |
                            set(df2["Category"].unique()) |
                            set(df3["Category"].unique()))
    summary = []

    for category in all_categories:
        urls1 = df1[df1["Category"] == category]
        urls2 = df2[df2["Category"] == category]
        urls3 = df3[df3["Category"] == category]

        urls1_set = set(urls1["Normalized URL"].dropna().unique())
        urls2_set = set(urls2["Normalized URL"].dropna().unique())
        urls3_set = set(urls3["Normalized URL"].dropna().unique())

        common_all = urls1_set & urls2_set & urls3_set

        only_in_1 = urls1_set - (urls2_set | urls3_set)
        only_in_2 = urls2_set - (urls1_set | urls3_set)
        only_in_3 = urls3_set - (urls1_set | urls2_set)

        total_unique = len(urls1_set | urls2_set | urls3_set)
        overlap_rate = len(common_all) / total_unique if total_unique > 0 else 0

        summary.append({
            "Category": category,
            "URLs in Day1": len(urls1_set),
            "URLs in Day2": len(urls2_set),
            "URLs in Day3": len(urls3_set),
            "Common URLs": len(common_all),
            "Overlap Rate": round(overlap_rate, 2)
        })

        print(f"\n📦 Category: {category}")
        print(f"✅ Common URLs (all three): {len(common_all)}")

        print(f"\n🔴 Only in {file1} ({len(only_in_1)}):")
        print(urls1[urls1["Normalized URL"].isin(only_in_1)]
              [["Source Link", "Source Rank"]]
              .drop_duplicates(subset="Source Link")
              .sort_values("Source Rank"))

        print(f"\n🔵 Only in {file2} ({len(only_in_2)}):")
        print(urls2[urls2["Normalized URL"].isin(only_in_2)]
              [["Source Link", "Source Rank"]]
              .drop_duplicates(subset="Source Link")
              .sort_values("Source Rank"))

        print(f"\n🟢 Only in {file3} ({len(only_in_3)}):")
        print(urls3[urls3["Normalized URL"].isin(only_in_3)]
              [["Source Link", "Source Rank"]]
              .drop_duplicates(subset="Source Link")
              .sort_values("Source Rank"))

        if plot:
            venn3([urls1_set, urls2_set, urls3_set],
                  set_labels=("Day 1", "Day 2", "Day 3"))
            plt.title(f"URL Overlap for {category}")
            plt.show()

    return pd.DataFrame(summary)

# --- Main Execution ---
if __name__ == "__main__":
    print("🟡 Running 2-day comparison...")
    result_2_day = compare_2_urls_by_category(
        "googlesearch_brand_3_31.csv",
        "googlesearch_brand_4_1.csv",
        plot=True
    )
    print("\n📊 Summary of 2-Day Overlap:")
    print(result_2_day.sort_values(by="Overlap Rate", ascending=False))

    print("\n🟢 Running 3-day comparison...")
    result_3_day = compare_google_urls_by_category_3(
        "GoogleSearch_Data_3_31.csv",
        "GoogleSearch_Data_4_1.csv",
        "GoogleSearch_Data_4_2.csv",
        plot=True
    )
    print("\n📊 Summary of 3-Day Overlap:")
    print(result_3_day.sort_values(by="Overlap Rate", ascending=False))
