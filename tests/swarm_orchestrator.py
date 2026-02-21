import requests
import time
import json

# =======================================================================
# AI Swarm Orchestrator
# Demonstrates how to route specific tasks to specific GPU Workers
# =======================================================================

COORDINATOR_URL = "http://localhost:8000/v1"

# Define your Swarm Agents
AGENTS = {
    "Coder": {
        "model": "deepseek-7b",
        "worker_id": "worker-gpu-0",  # E.g., The 3080 12GB
        "system_prompt": "You are a Senior Python Developer. Write clean, efficient code.",
    },
    "Reviewer": {
        "model": "deepseek-7b",
        "worker_id": "worker-gpu-1",  # E.g., The AMD 9060XT
        "system_prompt": "You are a strict Code Reviewer. Analyze code for bugs and improvements.",
    }
}

def ask_agent(agent_name: str, prompt: str) -> str:
    """Send a prompt to a specific agent/GPU via the Coordinator."""
    agent_config = AGENTS[agent_name]
    
    print(f"\n[{agent_name}] Thinking on {agent_config['worker_id']}...")
    
    payload = {
        "model": agent_config["model"],
        "prompt": f"{agent_config['system_prompt']}\n\nUser: {prompt}\nAssistant:",
        "worker_id": agent_config["worker_id"],
        "max_tokens": 512,
        "temperature": 0.3
    }

    start_time = time.time()
    try:
        response = requests.post(
            f"{COORDINATOR_URL}/completions",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        
        duration = time.time() - start_time
        print(f"[{agent_name}] Replied in {duration:.1f}s (worker: {result.get('worker_id')})")
        return result["text"].strip()
        
    except requests.exceptions.RequestException as e:
        print(f"[{agent_name}] Error connecting to Coordinator: {e}")
        return ""


if __name__ == "__main__":
    print("=== Starting Swarm Orchestrator ===")
    
    # 1. Ask the Coder to write something
    initial_task = "Write a Python script that calculates the Fibonacci sequence up to n."
    print(f"\n[Task] {initial_task}")
    
    code_output = ask_agent("Coder", initial_task)
    print(f"\n--- Output from Coder ---\n{code_output}\n-------------------------")
    
    if code_output:
        # 2. Take the Coder's exact output and send it to the Reviewer
        review_prompt = f"Please review this code and suggest improvements:\n\n{code_output}"
        
        review_output = ask_agent("Reviewer", review_prompt)
        print(f"\n--- Output from Reviewer ---\n{review_output}\n-------------------------")
