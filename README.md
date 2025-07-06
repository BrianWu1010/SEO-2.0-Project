# SEO-2.0-Project
This proposal outlines a strategy to boost Nestlé product visibility across generative AI search engines by analyzing ranking algorithms, testing SEO 2.0 tactics, and developing scalable optimization guidelines.

## Consultant_Report.pdf
The final consultant report, included in this repository as a PDF file, summarizes our full project findings and strategic recommendations for improving product visibility across generative AI search systems. It integrates both technical and business perspectives, drawing insights from the various notebooks and data analyses we conducted throughout the project. The report includes:

- A summary of experiments and methods
- Quantitative comparisons across ChatGPT-4o, Copilot, Gemini, and traditional search engines (Google, Bing)
- Analysis on Amazon Rufus
- Analysis of source diversity, ranking consistency, and citation patterns
- Practical suggestions for SEO 2.0 strategies
- Visuals and figures directly generated from our notebooks

This report serves as the primary deliverable for the SEO 2.0 project and is designed to be accessible to both technical stakeholders and non-technical business decision-makers.


## 🗂️ Personal Contribution — Jiamei Shu
Jiamei was responsible for the end-to-end analysis of traditional and LLM-based web search outputs, focusing on understanding how product recommendations are formed and ranked. Her contributions include:

- Built the scraping and analysis pipeline for collecting traditional search engine results (Google and Bing) using SerpAPI
- Conducted UI-based prompt experiments on ChatGPT-4o and Copilot, and manually compiled data for downstream comparison
- Designed ranking normalization and URL-level overlap methods to evaluate consistency across models and search platforms
- Developed Python scripts and Jupyter notebooks for data analysis, visualization, and comparative benchmarking
- Contributed figures and insights to the final consultant report, with a focus on source diversity, ranking behavior, and practical implications

### data/jiamei/
This folder contains raw and intermediate data used during my analysis. These datasets were collected during prompt-driven evaluation and traditional search engine benchmarking, and served as the foundation for comparing model outputs at both the URL and brand levels.

### notebooks/jiamei/
This folder contains the core Jupyter notebooks that I developed and maintained throughout the project. Each notebook tackles a different aspect of our evaluation and supports key conclusions presented in the final report. Descriptions are as follows:

- **ChatGPT4o_UI_Result_Analysis.ipynb**
This notebook analyzes product recommendation results from the ChatGPT-4o user interface. It standardizes ranking metrics across multiple runs (Rank_norm), explores the correlation between product rank and the number of source URLs, and visualizes reference distribution using boxplots and heatmaps. The notebook also includes a domain-level breakdown of URL concentration patterns and a comparison of reference distribution across ranks.

- **Brand_level_Comparison_between_Traditional_Search_and_LLM_Web_Search_API.ipynb**
This notebook extracts brand names from product titles using OpenAI's GPT-4o-mini model and compares the frequency of brand mentions across different search systems—Google Shopping, Google Search, and LLM web search APIs. The goal is to evaluate brand dominance, potential biases, and brand visibility across platforms. It includes preprocessing steps for harmonizing product data and visualization of brand-level overlaps and disparities.

- **Traditional_Search_Engine_Results_Collection.ipynb**
This notebook automates the retrieval of product search results from Google and Bing using the SerpAPI. It defines helper functions for query execution, normalizes URLs, and aggregates results across five predefined product categories. It supports benchmarking against LLM-based methods and serves as the ground truth reference for domain overlap comparison.

- **URL_Level_Comparison_between_Google_Search_and_Open_Perplex_API.ipynb**
This notebook compares URL-level output between Google Search and the Perplexity API. It focuses on understanding how the open-source Perplexity model aligns with traditional search engines in terms of referencing product pages. The analysis includes URL normalization, frequency counts, and overlap summaries across multiple categories.

- **URL_Level_Google_Search_Daily_Date_Comparison.ipynb**
This notebook analyzes the consistency of Google Search results over time. It compares search outputs from consecutive days and examines how the set of retrieved product URLs evolves. The analysis includes normalized URL processing, intersection tracking, and overlap rate visualization to highlight the volatility of search rankings on a daily basis.

- **URL_Level_Comparison_between_Traditional_Search_and_LLM_Web_Search_APIs.ipynb**
This notebook performs a three-way comparison between traditional search engines (Google/Bing) and LLM-powered web search APIs (e.g., OpenAI Web Search, Perplexity, Sonar Pro). It quantifies the overlap in referenced URLs by category, generates visual summaries with Venn diagrams, and helps assess the alignment between AI-generated recommendations and traditional search infrastructure.

Each notebook was iteratively developed based on project needs and contributed directly to our written insights and visualizations.

### src/jiamei/
This folder contains the .py source files used to support and automate portions of my analysis. It includes helper functions for web search result collection, URL normalization, overlap visualization, and formatted output saving for downstream use in plots and summary tables.

