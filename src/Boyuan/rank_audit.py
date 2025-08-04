import os
import json
import csv
import time
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import MessageRole, BingGroundingTool, ThreadMessage
from azure.identity import DefaultAzureCredential

load_dotenv()

# Azure setup
project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str=os.environ["PROJECT_CONNECTION_STRING"],
)
bing_connection = project_client.connections.get(
    connection_name=os.environ["BING_CONNECTION_NAME"]
)
bing = BingGroundingTool(connection_id=bing_connection.id)

# Polling constants
RUN_POLL_INTERVAL = 5    # seconds
MAX_RUN_DURATION = 120   # seconds per run

def _poll_run(thread_id: str, run_id: str):
    """Wait until the specified run succeeds or fails, up to MAX_RUN_DURATION."""
    start = time.time()
    run = project_client.agents.get_run(thread_id=thread_id, run_id=run_id)
    while run.status not in ("succeeded", "failed"):
        elapsed = time.time() - start
        print(f"[Polling] Run {run_id} status={run.status} elapsed={elapsed:.1f}s")
        if elapsed > MAX_RUN_DURATION:
            raise TimeoutError(f"Run {run_id} timed out after {MAX_RUN_DURATION} seconds.")
        time.sleep(RUN_POLL_INTERVAL)
        run = project_client.agents.get_run(thread_id=thread_id, run_id=run_id)
    if run.status == "failed":
        raise RuntimeError(f"Run {run_id} failed: {run.last_error}")
    print(f"[Polling] Run {run_id} completed with status={run.status} in {time.time()-start:.1f}s")
    return run


def _extract_text(msg: ThreadMessage) -> str:
    """Pull the first text reply from a ThreadMessage."""
    if msg.text_messages:
        return msg.text_messages[0].text.value.strip()
    return ""


def run_rank_audit(category: str):
    # Prompts
    prompt_ranking = f"""
Give me a ranked list of the top 10 popular **{category}** products currently sold in Canada.

Only include:
- Product Title
- Brand

List them in order from 1 to 10, no explanation.
"""

    prompt_reasoning = f"""
What criteria did you use to rank the top 10 {category} products?
Explain why you placed your top 3 choices at the top.
"""

    # Create a temporary agent
    agent = project_client.agents.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        name=f"rank-audit-agent-{category.replace(' ', '-')}",
        instructions=(
            "You are a product analyst that provides objective product rankings "
            "and reasoning using Bing-grounded information only."
        ),
        tools=bing.definitions,
        headers={"x-ms-enable-preview": "true"},
    )

    try:
        # ---- RUN 1: Ranking ----
        thread_rank = project_client.agents.create_thread()
        project_client.agents.create_message(
            thread_id=thread_rank.id,
            role=MessageRole.USER,
            content=prompt_ranking
        )
        run_rank = project_client.agents.create_and_process_run(
            thread_id=thread_rank.id,
            agent_id=agent.id
        )
        try:
            _poll_run(thread_rank.id, run_rank.id)
        except TimeoutError as e:
            print(f"[Warning] Ranking run timed out: {e}")
        # Retrieve whatever ranking messages arrived
        msgs_rank = list(project_client.agents.list_messages(
            thread_id=thread_rank.id, run_id=run_rank.id
        ))
        assistant_rank = next(
            (m for m in msgs_rank if isinstance(m, ThreadMessage) and m.role == MessageRole.ASSISTANT),
            None
        )
        ranking_text = _extract_text(assistant_rank) if assistant_rank else "[Error] No ranking reply received."
        print("\n[Intermediate Output] Ranking Result:\n", ranking_text)

        # ---- RUN 2: Reasoning ----
        thread_reason = project_client.agents.create_thread()
        project_client.agents.create_message(
            thread_id=thread_reason.id,
            role=MessageRole.USER,
            content=prompt_reasoning
        )
        run_reason = project_client.agents.create_and_process_run(
            thread_id=thread_reason.id,
            agent_id=agent.id
        )
        try:
            _poll_run(thread_reason.id, run_reason.id)
        except TimeoutError as e:
            print(f"[Warning] Reasoning run timed out: {e}")
        msgs_reason = list(project_client.agents.list_messages(
            thread_id=thread_reason.id, run_id=run_reason.id
        ))
        assistant_reason = next(
            (m for m in msgs_reason if isinstance(m, ThreadMessage) and m.role == MessageRole.ASSISTANT),
            None
        )
        explanation_text = _extract_text(assistant_reason) if assistant_reason else "[Error] No reasoning reply received."
        print("\n[Intermediate Output] Reasoning Result:\n", explanation_text)

    finally:
        # Clean up the agent
        project_client.agents.delete_agent(agent.id)

    # Final outputs
    print("\nTop 10 Products:\n", ranking_text)
    print("\nExplanation:\n", explanation_text)

    # Parse the ranking lines
    products = []
    for line in ranking_text.splitlines():
        if line and line[0].isdigit():
            parts = line.split(". ", 1)
            rank = parts[0].strip()
            title_brand = parts[1] if len(parts) > 1 else ""
            if " - " in title_brand:
                name, brand = title_brand.split(" - ", 1)
            else:
                name, brand = title_brand, ""
            products.append({"Rank": rank, "Product Title": name.strip(), "Brand": brand.strip()})

    # Save to JSON and CSV
    base = category.replace(" ", "_").lower()
    with open(f"{base}_rank_audit.json", "w", encoding="utf-8") as f:
        json.dump({"ranking": products, "reasoning": explanation_text}, f, indent=2, ensure_ascii=False)
    with open(f"{base}_rank_audit.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Rank", "Product Title", "Brand"])
        writer.writeheader()
        writer.writerows(products)

    print(f"\nSaved to {base}_rank_audit.json and {base}_rank_audit.csv")


if __name__ == "__main__":
    run_rank_audit("Instant Coffee")
