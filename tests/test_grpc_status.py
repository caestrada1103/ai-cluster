import grpc
import cluster_pb2 as pb
import cluster_pb2_grpc as pb_grpc
import asyncio

async def test_worker_status():
    channel = grpc.aio.insecure_channel('localhost:50051')
    stub = pb_grpc.WorkerStub(channel)
    try:
        status = await stub.GetStatus(pb.Empty(), timeout=5)
        print("Worker Status:")
        print(f"Worker ID: {status.worker_id}")
        print(f"GPUs: {status.gpus}")
        print(f"Len GPUs: {len(status.gpus)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_worker_status())
