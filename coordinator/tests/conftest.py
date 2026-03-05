"""pytest configuration for coordinator tests.

The generated gRPC stubs in coordinator/proto/ use bare (non-package-qualified)
imports (e.g. `import cluster_pb2`). Add the proto directory to sys.path so
those imports resolve correctly during test collection.
"""

import sys
from pathlib import Path

# coordinator/proto/ must be on sys.path before any coordinator.* import
# that transitively pulls in coordinator.proto.cluster_pb2_grpc
sys.path.insert(0, str(Path(__file__).parent.parent / "proto"))
