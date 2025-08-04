import os
import re
import json
import csv
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import MessageRole, BingGroundingTool
from azure.identity import DefaultAzureCredential

# Load .env variables
load_dotenv()

# Setup Azure AI project
project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str=os.environ["PROJECT_CONNECTION_STRING"],
)
bing_connection = project_client.connections.get(connection_name=os.environ["BING_CONNECTION_NAME"])
bing = BingGroundingTool(connection_id=bing_connection.id)

# Category and prompt
category = "Baking Ingredients"
prompt = f"""
Find 10 real, popular **baking ingredient products** currently sold in Canada.

For each product, include:

- Product Title: full name of the product
- Description: what it’s used for (e.g. baking cookies, cakes, etc.)
- Brand: name of the brand or company
- Retailer: where it's sold (Amazon, Walmart, etc.)
- Source URL: direct link to the product
- Comments: anything interesting, like reviews, popularity, price

Only include real products with working URLs. Search Canadian shopping websites.
Return results only in the requested format.
"""

# Run the query
def run_single_query(prompt):
    print("🔄 Sending prompt to GPT-4o + Bing...\n")
    agent = project_client.agents.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        name="debug-agent",
        instructions="Return product data using structured format with Bing support. Do not include any text other than the product list.",
        tools=bing.definitions,
        headers={"x-ms-enable-preview": "true"},
    )
    thread = project_client.agents.create_thread()
    project_client.agents.create_message(thread_id=thread.id, role=MessageRole.USER, content=prompt)
    run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent.id)

    if run.status == "failed":
        print(f"❌ Agent failed: {run.last_error}")
        return None

    response = project_client.agents.list_messages(thread_id=thread.id).get_last_message_by_role(MessageRole.AGENT)
    content = "\n".join([msg.text.value for msg in response.text_messages])
    project_client.agents.delete_agent(agent.id)
    
    print("📝 RAW LLM OUTPUT:\n", content[:800])  # print first 800 chars
    return content

# Extract structured product data
def parse_products(text):
    product_info = []
    blocks = text.strip().split("\n\n")
    print(f"\n🔍 Found {len(blocks)} product blocks")

    for block in blocks:
        print("\n🔹 BLOCK:\n", block)
        product = {
        'Product Title': re.search(r"Product Title:\s*(.*)", block).group(1) if re.search(r"Product Title:\s*(.*)", block) else "",
        'Description': re.search(r"Description:\s*(.*)", block).group(1) if re.search(r"Description:\s*(.*)", block) else "",
        'Brand': re.search(r"Brand:\s*(.*)", block).group(1) if re.search(r"Brand:\s*(.*)", block) else "",
        'Retailer': re.search(r"Retailer:\s*(.*)", block).group(1) if re.search(r"Retailer:\s*(.*)", block) else "",
        'Source URL': re.search(r"Source URL:\s*(http[s]?://\S+)", block).group(1) if re.search(r"Source URL:\s*(http[s]?://\S+)", block) else "",
        'Comments': re.search(r"Comments:\s*(.*)", block).group(1) if re.search(r"Comments:\s*(.*)", block) else "",
        }

        if any(product.values()):
            product_info.append(product)
    print(f"\n✅ Parsed {len(product_info)} products.")
    return product_info

# Save as JSON and CSV
def save_outputs(data, base_filename):
    json_path = f"{base_filename}.json"
    csv_path = f"{base_filename}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

    print(f"💾 Saved to {json_path} and {csv_path}")

# MAIN
if __name__ == "__main__":
    raw_output = run_single_query(prompt)
    if raw_output:
        parsed = parse_products(raw_output)
        if parsed:
            save_outputs(parsed, "test_baking_ingredients")
        else:
            print("❌ Parsing failed — no products extracted.")
    else:
        print("❌ No output returned from GPT.")
