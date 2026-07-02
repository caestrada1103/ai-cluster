import coordinator.proto.cluster_pb2 as pb

request = pb.LoadModelRequest(
    model_name="test",
    model_path="",
    gpu_ids=[0]
)

print(f"Request: {request}")
print(f"gpu_ids len: {len(request.gpu_ids)}")
