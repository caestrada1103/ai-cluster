import sys
import os
import grpc
import time

# Add coordinator path (one level up from tests/)
# Assuming this script is in ./tests/ and coordinator is in ./coordinator
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
coordinator_path = os.path.join(project_root, "coordinator")

sys.path.append(coordinator_path)
proto_path = os.path.join(coordinator_path, "proto")
sys.path.append(proto_path)

try:
    import cluster_pb2
    import cluster_pb2_grpc
except ImportError:
    print(f"Error importing protos. Paths: {sys.path}")
    raise

def run():
    print("Connecting to worker at localhost:50051...")
    channel = grpc.insecure_channel('localhost:50051')
    stub = cluster_pb2_grpc.WorkerStub(channel)

    print("Checking health...")
    try:
        response = stub.HealthCheck(cluster_pb2.Empty())
        print(f"Health Status: {response.status} (1=SERVING)")
    except grpc.RpcError as e:
        print(f"Failed to connect: {e}")
        return

    # Use a small model for testing
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # model_name = "deepseek-7b"
    print(f"\nLoading model {model_name}...")
    print("(This may take a while to download weights on first run...)")
    
    try:
        req = cluster_pb2.LoadModelRequest(
            model_name=model_name,
            quantization=cluster_pb2.Quantization.FP16
            # config is empty -> worker downloads from HF
        )
        # Increase timeout for loading (download takes time)
        resp = stub.LoadModel(req, timeout=600)
        print(f"Load Response: Success={resp.success}, Msg='{resp.message}'")
        print(f"Memory Used: {resp.memory_used / 1024 / 1024:.2f} MB")
        
        if not resp.success:
            print("Model load failed!")
            return

    except grpc.RpcError as e:
        print(f"Load failed: {e}")
        return

    print(f"\nRunning inference on {model_name}...")
    prompt = "The rust programming language is"
    print(f"Prompt: '{prompt}'")
    
    try:
        req = cluster_pb2.InferenceRequest(
            model_name=model_name,
            prompt=prompt,
            max_tokens=30,
            temperature=0.7,
            top_p=0.9,
            top_k=40
        )
        
        print("Response:", end=" ", flush=True)
        for resp in stub.Infer(req):
            print(resp.text, end="", flush=True)
        print("\n\nInference complete.")
        
    except grpc.RpcError as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    run()
