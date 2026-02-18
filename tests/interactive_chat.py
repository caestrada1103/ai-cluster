import sys
import os
import time
import grpc

# Add coordinator/proto to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'coordinator', 'proto'))

import cluster_pb2
import cluster_pb2_grpc

def chat():
    # Connect to worker
    channel = grpc.insecure_channel('localhost:50051')
    stub = cluster_pb2_grpc.WorkerStub(channel)

    print("Connecting to worker...")
    try:
        # Check health
        response = stub.HealthCheck(cluster_pb2.Empty())
        if response.status != cluster_pb2.HealthCheckResponse.SERVING:
            print("Worker is not serving!")
            return
            
        print("Worker is ready!")
    except grpc.RpcError as e:
        print(f"Could not connect to worker: {e.details()}")
        return

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Ensure model is loaded (optional if already loaded, but good practice)
    # simple check or just try load (it's idempotentish in our logic? No, load_model checks cache)
    print(f"Ensuring model {model_name} is loaded...")
    try:
        stub.LoadModel(cluster_pb2.LoadModelRequest(
            model_name=model_name,
            quantization=cluster_pb2.Quantization.FP16
        ))
    except grpc.RpcError as e:
        print(f"Load failed: {e}")
        # convert to string to check if it says "already loaded" or similar?
        # Our worker logs "Model ... already loaded" and returns OK. 
        # So grpc call should succeed even if loaded.

    print("\n" + "="*50)
    print(f"Chatting with {model_name}")
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
            
            # Simple inference
            # To make it chat-like, we might want to format prompt? 
            # TinyLlama Chat format: <|system|>\n...<|user|>\n...<|assistant|>\n
            # For now, raw prompt is fine for testing.
            
            full_prompt = f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
            
            request = cluster_pb2.InferenceRequest(
                model_name=model_name,
                prompt=full_prompt,
                max_tokens=100,
                temperature=0.7,
                stream=False 
            )
            
            response_stream = stub.Infer(request)
            
            # If stream=False, we get one response? 
            # The proto definition says `rpc Infer(...) returns (stream InferenceResponse);`
            # Even if stream=False, it returns a stream of 1 item? 
            # Or worker implementation streams tokens?
            # Worker implementation respects `stream` flag.
            # If stream=True, it yields tokens.
            # If stream=False, it yields one response with full text? 
            # Let's check worker.rs.
            
            # It seems our worker implementation ALWAYS returns a stream.
            # `type ResponseStream = Pin<Box<dyn Stream<Item = Result<InferenceResponse, Status>> + Send>>;`
            
            full_text = ""
            for response in response_stream:
                print(response.text, end="", flush=True)
                full_text += response.text
                
            print("\n")
            
        except KeyboardInterrupt:
            break
        except grpc.RpcError as e:
            print(f"\nRPC Error: {e.details()}")

if __name__ == "__main__":
    chat()
