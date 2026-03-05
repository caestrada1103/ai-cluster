import sys
import os
import time
import grpc

import cluster_pb2
import cluster_pb2_grpc

import argparse

def chat():
    parser = argparse.ArgumentParser(description='Interactive chat with AI Worker')
    parser.add_argument('--max-tokens', type=int, default=512, help='Maximum tokens to generate (default: 512)')
    parser.add_argument('--temp', type=float, default=0.7, help='Temperature (default: 0.7)')
    parser.add_argument('--host', type=str, default='localhost:50051', help='Worker gRPC host')
    parser.add_argument('--quant', type=str, default='fp16', choices=['fp16', 'int8', 'int4'], help='Quantization type (default: fp16)')
    args = parser.parse_args()

    # Map string to proto enum
    quant_map = {
        'fp16': cluster_pb2.Quantization.FP16,
        'int8': cluster_pb2.Quantization.INT8,
        'int4': cluster_pb2.Quantization.INT4,
    }
    quantization = quant_map.get(args.quant, cluster_pb2.Quantization.FP16)

    # Connect to worker
    channel = grpc.insecure_channel(args.host)
    stub = cluster_pb2_grpc.WorkerStub(channel)

    print(f"Connecting to worker at {args.host}...")
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
    # model_name = "deepseek-7b"
    
    # Ensure model is loaded (optional if already loaded, but good practice)
    print(f"Ensuring model {model_name} is loaded with {args.quant} quantization...")
    try:
        stub.LoadModel(cluster_pb2.LoadModelRequest(
            model_name=model_name,
            quantization=quantization
        ))
    except grpc.RpcError as e:
        print(f"Load failed: {e}")

    print("\n" + "="*50)
    print(f"Chatting with {model_name}")
    print(f"Params: max_tokens={args.max_tokens}, temp={args.temp}")
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
            
            request = cluster_pb2.InferenceRequest(
                model_name=model_name,
                prompt=full_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temp,
                stream=True 
            )
            
            response_stream = stub.Infer(request)
            
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
