import argparse
import requests
import sys

def chat():
    parser = argparse.ArgumentParser(description='Cluster Chat via HTTP API')
    parser.add_argument('--host', type=str, default='localhost:8000', help='Coordinator host:port')
    parser.add_argument('--model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0', help='Model name')
    parser.add_argument('--max-tokens', type=int, default=512, help='Max tokens')
    parser.add_argument('--temp', type=float, default=0.7, help='Temperature')
    args = parser.parse_args()

    # Ensure http protocol
    if not args.host.startswith("http"):
        base_url = f"http://{args.host}"
    else:
        base_url = args.host
        
    print(f"Connecting to Cluster Coordinator at {base_url}...")

    # Check health / docs
    try:
        resp = requests.get(f"{base_url}/docs", timeout=5)
        if resp.status_code == 200:
            print("Coordinator is online!")
        else:
            print(f"Coordinator reachable but returned {resp.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Could not connect to Coordinator: {e}")
        return

    # Check workers
    try:
        resp = requests.get(f"{base_url}/v1/workers", timeout=5)
        # Note: API routes are mounted at /v1 usually. Checking...
        # api.py docstring says: "mounted at /v1 in main.py".
        # So it's /v1/workers.
        # But wait, main.py snippet didn't show the mount path explicitly?
        # It showed `app.include_router(api_router, prefix="/v1")` usually.
        # Let's assume /v1 or try root.
        
        # If /v1/workers fails, try /workers.
        if resp.status_code != 200:
             resp = requests.get(f"{base_url}/workers", timeout=5)
             
        if resp.status_code == 200:
            workers = resp.json()
            print(f"Found {len(workers)} active workers.")
            for w in workers:
                 # api.py WorkerInfoResponse schema: id, address, state...
                 print(f" - {w.get('id', 'unknown')} at {w.get('address')} ({w.get('state')})")
        else:
             print("Failed to fetch workers list.")
    except Exception as e:
        print(f"Worker check failed: {e}")

    print("\n" + "="*50)
    print(f"Chatting with {args.model} via Cluster")
    print("Type 'quit' or 'exit' to stop.")
    print("="*50 + "\n")

    while True:
        try:
            prompt = input("You: ")
            if prompt.lower() in ['quit', 'exit']:
                break
            
            if not prompt.strip():
                continue

            print("Model: ", end="", flush=True)
            
            full_prompt = f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
            
            payload = {
                "model": args.model,
                "prompt": full_prompt,
                "max_tokens": args.max_tokens,
                "temperature": args.temp,
                "stream": False # Streaming requires SSE client logic
            }
            
            # Try /v1/completions first
            endpoint = f"{base_url}/v1/completions"
            resp = requests.post(endpoint, json=payload, timeout=60)
            
            if resp.status_code == 404:
                # Try root /completions
                endpoint = f"{base_url}/completions"
                resp = requests.post(endpoint, json=payload, timeout=60)

            if resp.status_code == 200:
                data = resp.json()
                # api.py CompletionResponse schema: text, ...
                print(data.get("text", ""))
            else:
                 print(f"Error {resp.status_code}: {resp.text}")

            print("\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    chat()
