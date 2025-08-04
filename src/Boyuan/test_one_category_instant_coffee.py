import os
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

system_prompt = f"""
You are an intelligent data extraction agent.

Extract structured product information from blog text related to the '{category}' category. 
Follow this schema exactly for each product you find:

class Product:
    brand: str = "The brand of the product, e.g. 'Nestle'"
    long_title: str = "The full product name, e.g. 'Nescafe Classic Instant Coffee'"
    short_title: str = "A short version of the product name, e.g. 'Nescafe Classic'"
    description: str = "One sentence that best describes this product."
    rank: int = "The integer rank of the product in the blog post"
    urls: List[str] = "A list of all reference URLs where the product is recommended."

Make sure the output is a valid JSON list of Product objects.
"""

# Run the query
def run_single_query(prompt):
    print("Sending prompt to GPT-4o + Bing...\n")
    agent = project_client.agents.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        name="debug-agent",
        instructions="Return only a JSON list of Product objects. Do not include explanatory text. Use the provided schema.",
        tools=bing.definitions,
        headers={"x-ms-enable-preview": "true"},
    )
    thread = project_client.agents.create_thread()
    project_client.agents.create_message(thread_id=thread.id, role=MessageRole.USER, content=prompt)
    run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent.id)

    if run.status == "failed":
        print(f"Agent failed: {run.last_error}")
        return None

    response = project_client.agents.list_messages(thread_id=thread.id).get_last_message_by_role(MessageRole.AGENT)
    content = "\n".join([msg.text.value for msg in response.text_messages])
    project_client.agents.delete_agent(agent.id)
    
    print("RAW LLM OUTPUT (truncated):\n", content[:800])
    return content

# Parse valid JSON product list
def parse_products(text):
    try:
        products = json.loads(text)
        print(f"Parsed {len(products)} products from JSON.")
        return products
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
        return []

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

    print(f"Saved to {json_path} and {csv_path}")

# MAIN
if __name__ == "__main__":
    raw_output = run_single_query(system_prompt)
    if raw_output:
        parsed = parse_products(raw_output)
        if parsed:
            save_outputs(parsed, "test_baking_ingredients")
        else:
            print("Parsing failed — no products extracted.")
    else:
        print("No output returned from GPT.")
