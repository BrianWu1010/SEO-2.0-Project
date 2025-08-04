import os
import re
import json
import csv
import time
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import MessageRole, BingGroundingTool
from azure.identity import DefaultAzureCredential

# Load environment
load_dotenv()

# Azure setup
project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str=os.environ["PROJECT_CONNECTION_STRING"],
)
bing_connection = project_client.connections.get(connection_name=os.environ["BING_CONNECTION_NAME"])
bing = BingGroundingTool(connection_id=bing_connection.id)

# Categories to loop
categories = [
    "Baking Ingredients",
    "Instant Coffee",
    "Dog Food",
    "Cat Food",
    "Cat Litter"
]

# Function to call GPT-4o + Bing
def grounded_query(category: str, retry=3):
    prompt = f"""
    Search Bing and return 10 real products currently available in Canada under the category **{category}**.
    Each product must include:
        - Product Title
        - Brand
        - Source URL (only if a real product link is found)
    Only return results in the format below with no introduction or explanation.

    Format each product block with a blank line between them.
    """

    for attempt in range(retry):
        try:
            agent = project_client.agents.create_agent(
                model=os.environ["MODEL_DEPLOYMENT_NAME"],
                name=f"agent-{category.replace(' ', '-')}",
                instructions="Return product data using structured format with Bing support only. No intro or summary.",
                tools=bing.definitions,
                headers={"x-ms-enable-preview": "true"},
            )

            thread = project_client.agents.create_thread()
            project_client.agents.create_message(
                thread_id=thread.id,
                role=MessageRole.USER,
                content=prompt
            )

            run = project_client.agents.create_and_process_run(
                thread_id=thread.id,
                agent_id=agent.id
            )

            error_code = getattr(run.last_error, "code", "")
            if run.status == "failed":
                print(f"❌ Agent failed for {category}: {run.last_error}")
                if error_code == 'rate_limit_exceeded':
                    wait_time = 60 + attempt * 10
                    print(f"⏳ Waiting {wait_time} seconds before retry (attempt {attempt+1}/{retry})...")
                    time.sleep(wait_time)
                    continue
                return None

            response = project_client.agents.list_messages(
                thread_id=thread.id
            ).get_last_message_by_role(MessageRole.AGENT)

            content = "\n".join([msg.text.value for msg in response.text_messages])
            project_client.agents.delete_agent(agent.id)
            return content

        except Exception as e:
            print(f"⚠️ Unexpected error during attempt {attempt+1}/{retry}: {e}")
            time.sleep(30)
            continue

    print(f"🚫 Giving up on category: {category} after {retry} attempts.")
    return None


# Product parser
def extract_product_info(response_text):
    product_info = []
    blocks = response_text.strip().split("\n\n")
    print(f"🔍 Found {len(blocks)} blocks")

    for block in blocks:
        print("🔹 Block:\n", block)
        product = {
            'Product Title': re.search(r"Product Title:\s*(.*)", block).group(1) if re.search(r"Product Title:\s*(.*)", block) else "",
            'Brand': re.search(r"Brand:\s*(.*)", block).group(1) if re.search(r"Brand:\s*(.*)", block) else "",
            'Source URL': re.search(r"Source URL:\s*(http[s]?://\S+)", block).group(1) if re.search(r"Source URL:\s*(http[s]?://\S+)", block) else "",
        }
        if any(product.values()):
            product_info.append(product)

    return product_info


# Save data
def save_outputs(data, category):
    base = category.replace(" ", "_").lower()
    json_path = f"{base}_products.json"
    csv_path = f"{base}_products.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

    print(f"✅ Saved to {json_path} and {csv_path}")

# MAIN LOOP
if __name__ == "__main__":
    for category in categories:
        print(f"\n🔄 Processing category: {category}")
        raw = grounded_query(category)
        if raw:
            print("\n🧠 Raw GPT Output Preview:")
            print(raw[:800])
            parsed = extract_product_info(raw)
            if parsed:
                save_outputs(parsed, category)
            else:
                print(f"❌ Parsing failed for: {category}")
        else:
            print(f"❌ No response for: {category}")
        time.sleep(5)  # small delay to reduce API pressure
