"""
Preparation
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from urllib.parse import urlparse

# Load and process ChatGPT4o UI data
ChatGPT4o_1 = pd.read_csv("ChatGPT4o and Copilot Data - Instant Coffee - ChatGPT4o - Boyuan.csv")
ChatGPT4o_2 = pd.read_csv("ChatGPT4o and Copilot Data - Instant Coffee - ChatGPT4o - Jiamei.csv")
ChatGPT4o_3 = pd.read_csv("ChatGPT4o and Copilot Data - Instant Coffee - ChatGPT4o - Wensha.csv")

ChatGPT4o_1['Ranking'] = ChatGPT4o_1.groupby('Run').cumcount() + 1
ChatGPT4o_2['Ranking'] = ChatGPT4o_2.groupby('Run').cumcount() + 1
ChatGPT4o_3['Ranking'] = ChatGPT4o_3.groupby('Run').cumcount() + 1

ChatGPT4o = pd.concat([ChatGPT4o_1, ChatGPT4o_2, ChatGPT4o_3])
ChatGPT4o = ChatGPT4o.drop(['Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16'], axis=1)

url_columns = [f'URL{i}' for i in range(1, 11)]
ChatGPT4o['URL_count'] = ChatGPT4o[url_columns].notna().sum(axis=1)
ChatGPT4o['id'] = range(1, len(ChatGPT4o) + 1)

print("Loaded ChatGPT4o data:")
print(ChatGPT4o.head())

"""
URL_count versus Ranking
"""
ChatGPT4o_noURLs = ChatGPT4o.drop(columns=url_columns)
print("\nURL-stripped Data Sample:")
print(ChatGPT4o_noURLs.head())

print("\nBasic Statistics:")
print(ChatGPT4o_noURLs[['Ranking', 'URL_count']].describe())

# Standardize rankings
ChatGPT4o_noURLs['Rank_norm'] = ChatGPT4o_noURLs.groupby('Run')['Ranking'].transform(lambda x: x / x.max())

# Boxplot visualization
plt.figure(figsize=(12, 6))
ax = sns.boxplot(x='Rank_norm', y='URL_count', data=ChatGPT4o_noURLs)

xticks = ax.get_xticks()
xticklabels = ax.get_xticklabels()

skip = 10
ax.set_xticks(xticks[::skip])
ax.set_xticklabels([label.get_text() for label in xticklabels[::skip]], rotation=45)

plt.title('URL Count across Rank_norm')
plt.xlabel('Normalized Ranking')
plt.ylabel('URL Count')
plt.tight_layout()
plt.show()

# Binned averages
ChatGPT4o_noURLs['Rank_bin'] = pd.qcut(ChatGPT4o_noURLs['Rank_norm'], q=5)
bin_avg = ChatGPT4o_noURLs.groupby('Rank_bin')['URL_count'].mean()
print("\nAverage URL Count by Rank Bin:")
print(bin_avg)

# Correlation
correlation = ChatGPT4o_noURLs[['Rank_norm', 'URL_count']].corr()
print("\nCorrelation Matrix:")
print(correlation)

"""
URLs versus Ranking
"""
def simplify_url(url):
    if pd.isna(url):
        return None
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"

for col in url_columns:
    ChatGPT4o[col] = ChatGPT4o[col].apply(simplify_url)

ChatGPT4o['id'] = range(len(ChatGPT4o))

url_long = ChatGPT4o.melt(
    id_vars=['id', 'Ranking'], value_vars=url_columns,
    var_name='URL_slot', value_name='Domain'
).dropna(subset=['Domain'])

domain_rank_counts = url_long.groupby(['Domain', 'Ranking']).size().unstack(fill_value=0)
print("\nDomain Rank Counts:")
print(domain_rank_counts)

# Heatmap for top domains
top_domains = url_long['Domain'].value_counts().head(5).index
domain_rank_matrix = url_long[url_long['Domain'].isin(top_domains)].groupby(['Domain', 'Ranking']).size().unstack(fill_value=0)
domain_rank_pct = domain_rank_matrix.div(domain_rank_matrix.sum(axis=1), axis=0) * 100
annot_labels = domain_rank_pct.applymap(lambda x: f"{x:.1f}%" if x > 0 else "")

plt.figure(figsize=(12, 6))
sns.heatmap(domain_rank_pct, annot=annot_labels, fmt="", cmap="Blues", cbar_kws={'label': 'Percentage (%)'})
plt.title("Domain Appearance Percentage by Ranking (Row-normalized)")
plt.xlabel("Ranking")
plt.ylabel("Domain")
plt.tight_layout()
plt.show()

"""
Comparison with traditional search engine
"""

# ⚠️ NOTE: Ensure this function is defined somewhere before use!
# from your_module import save_multiple_queries_to_csv

queries = {
    "Instant Coffee": "Top recommended Instant Coffee in Canada"
}

# Uncomment below if function is defined:
# save_multiple_queries_to_csv(queries_by_category=queries, engine="google", num_results=20, output_filename="serpapi_google_4_30.csv")
# save_multiple_queries_to_csv(queries_by_category=queries, engine="bing", num_results=20, output_filename="serpapi_bing_4_30.csv")

serpapi_bing = pd.read_csv("serpapi_bing_4_30.csv")
serpapi_google = pd.read_csv("serpapi_google_4_30.csv")

def get_match_stats(df, source_col, rank_col, domain_list):
    result = []
    for domain in domain_list:
        mask = df[source_col].str.contains(domain, na=False)
        matched_rows = df[mask]
        count = len(matched_rows)
        rank = matched_rows[rank_col].min() if count > 0 else None
        result.append((count, rank))
    return result

top_domains = [
    'https://www.allrecipes.com',
    'https://cornercoffeestore.com',
    'https://www.seriouseats.com',
    'https://www.bestproductscanada.com',
    'https://www.dropmocha.ca'
]

bing_stats = get_match_stats(serpapi_bing, 'Source', 'Source Rank', top_domains)
google_stats = get_match_stats(serpapi_google, 'Source', 'Source Rank', top_domains)

summary = pd.DataFrame({
    'Domain': top_domains,
    'Matched in Bing': [b[0] for b in bing_stats],
    'Source Rank in Bing': [b[1] for b in bing_stats],
    'Matched in Google': [g[0] for g in google_stats],
    'Source Rank in Google': [g[1] for g in google_stats]
})

print("\nSearch Engine Domain Match Summary:")
print(summary)

if __name__ == "__main__":
    print("\nScript executed successfully.")
