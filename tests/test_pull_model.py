import requests
import sys
import json
import time

COORDINATOR_URL = "http://localhost:8000/v1"

def register_worker():
    print("Registering worker locally...")
    response = requests.post(f"{COORDINATOR_URL}/workers/manual", json=["aicluster-worker-1:50051"])
    response.raise_for_status()
    results = response.json()["results"]
    if not results or results[0]["status"] != "connected":
        print(f"Failed to connect worker: {results}")
        sys.exit(1)
    
    worker_id = results[0]["id"]
    print(f"Successfully registered worker with ID: {worker_id}")
    return worker_id

def test_chat(worker_id):
    model_name = f"TinyLlama/TinyLlama-1.1B-Chat-v1.0@{worker_id}"
    # model_name = f"deepseek-7b@{worker_id}"
    print(f"\nSending chat completion request for model: {model_name}")
    print("This might take a minute as the Rust worker automatically downloads the model weights from HuggingFace...")
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{COORDINATOR_URL}/chat/completions",
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful AI"},
                    {"role": "user", "content": "Hello! What is your name?"}
                ],
                "max_tokens": 50,
                "temperature": 0.7
            },
            timeout=300 # 5 minutes for HF download
        )
        response.raise_for_status()
        
        data = response.json()
        end_time = time.time()
        print(f"\nTime taken: {end_time - start_time:.2f} seconds")
        print("Response received from UI proxy:")
        print(json.dumps(data, indent=2))
        
    except requests.exceptions.RequestException as e:
        print(f"\nChat completion failed: {e}")
        if e.response:
            print(f"Server response logic: {e.response.text}")

if __name__ == "__main__":
    worker_id = register_worker()
    
    # Check if registered properly:
    workers = requests.get(f"{COORDINATOR_URL}/workers").json()
    print("\nCurrent Workers in Registry:")
    print(json.dumps(workers, indent=2))
    
    test_chat(worker_id)
