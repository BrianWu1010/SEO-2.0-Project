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


## 🗂️ Personal Contribution — Wensha Sun
Wensha was responsible for integrating and orchestrating various LLM (Large Language Model) services to help extract meaningful insights from user queries. Using LangChain as the framework, Wensha connected to APIs from OpenAI, Perplexity, Openperplex, Hugging Face, Sonar and Sonar Pro to generate responses based on carefully designed prompts. The main goal was to explore brand popularity within specific product categories by analyzing which brands were most frequently recommended by LLMs in response to typical consumer questions.

- Integrated multiple LLM APIs (OpenAI, Perplexity, Openperplex, Hugging Face, Sonar, Sonar Pro) using LangChain.
- Engineered domain-specific prompts to extract brand recommendations across different categories.
- Developed automated pipelines to query LLMs and parse the results.
- Focused on five product categories relevant to Nestlé: cat food, dog food, cat litter, coffee, and baking ingredients. Analyzed and visualized which brands were most frequently mentioned by LLMs, including the ranking order in their responses (e.g., when LLMs list “1. Brand A 2. Brand B,” the position indicates perceived recommendation strength).

### data/wensha/
The data/ directory contains the raw output collected from various large language models (LLMs) during web search and product recommendation experiments. These outputs represent unprocessed responses from models such as ChatGPT (OpenAI), Perplexity AI, and HuggingFace-hosted models, used in response to structured prompts about consumer products.

The purpose of this raw data is to support comparative analysis, consistency checks, and source quality evaluations across different LLMs. Future steps in the workflow involve parsing, cleaning, and scoring this data for model benchmarking and downstream applications.

### notebooks/wensha/
The code focuses on evaluating and comparing the performance of various LLMs (Language Learning Models) — specifically OpenAI's ChatGPT, Perplexity AI, and HuggingFace models — in the context of web search and product recommendation tasks. 
Descriptions of the project across its key functional areas are as follows:

- **data_collect.py**
- **LLM-Based Web Search and Comparison**
The project initiates by setting up three distinct large language models — OpenAI’s ChatGPT, Perplexity AI, and HuggingFace's hosted models — to perform structured web searches.
Using LangChain integrations, the notebook standardizes inputs and outputs across all models. This allows the same prompt to be used with each model for fair comparison.
The objective is to evaluate how effectively each model retrieves useful and context-rich URLs when prompted to find information about specific consumer products.

- **Structured Product Data Collection**
A core part of the project involves defining a schema (Product class) for structuring information about products. Each product entry contains attributes like brand, short and long titles, a one-line description, and a ranked list of web sources with justifications for product recommendations. This structure ensures consistent formatting across LLM outputs and facilitates downstream processing or analysis. The use of typed pydantic models helps validate data collected from multiple models, ensuring it meets expectations for web search reliability and relevance.

- **Result Consistency and Quality Analysis**
After collecting the product-related data from each LLM, the notebook performs a consistency analysis across different models. This task evaluates whether the outputs are reproducible and coherent for the same prompt over multiple runs. The goal here is to identify models that provide stable, trustworthy recommendations — a crucial factor for tasks like market research, affiliate content creation, or product curation.

- **Comparative Analysis of Web Sources**
The final section centers around analyzing the quality of web sources returned by the different LLMs.
The notebook checks whether the URLs provided lead to authoritative, content-rich pages that genuinely explain why a product is recommended. This qualitative filter aims to weed out superficial or ad-focused content, promoting more trustworthy outputs. Such an evaluation is vital for use cases that depend on content sourcing, such as e-commerce content generation or SEO-oriented research.

- **API Key Management and Model Orchestration**
The notebook also includes secure management of API keys and model credentials, using getpass to prompt for sensitive data like OpenAI, Perplexity, and HuggingFace tokens. This enables seamless model switching and orchestrated querying across platforms. This orchestration layer acts as the foundation for consistent and reusable evaluation workflows.

- **Prompt Engineering and Output Parsing**
Finally, the notebook integrates PromptTemplate and multiple output parsers (Markdown and JSON) to manage how prompts are formatted and how outputs are interpreted. This allows the system to flexibly adapt the same logical task to different LLMs and retrieve structured results in the desired format, improving both model compatibility and downstream processing potential.
