import grpc
from cluster_pb2_grpc import WorkerServiceStub
from cluster_pb2 import RegisterWorkerRequest,
RegisterWorkerResponse

class GrpcWorkerClient:
    """Thin wrapper around the gRPC client used by the
coordinator."""

    def __init__(self, address: str = "localhost:50051"):
        self.address = address
        self.channel = grpc.aio.insecure_channel(self.address)
        self.stub = WorkerServiceStub(self.channel)

    async def register(self, worker_id: str, host: str, port: int):
        """Ask the worker to register itself with the
coordinator."""
        req = RegisterWorkerRequest(worker_id=worker_id, host=host,
port=port)
        await self.stub.Register(req)

    async def get_status(self):
        """Return simple status from the worker."""
        from cluster_pb2 import Empty
        resp = await self.stub.GetStatus(Empty())
        return resp