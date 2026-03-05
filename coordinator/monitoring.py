"""Monitoring and metrics module for the coordinator.

Provides a thin wrapper around prometheus_client to offer a consistent
metrics API used across coordinator modules.
"""

import logging
from typing import List, Optional

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class MetricsManager:
    """Central metrics manager providing factory methods for Prometheus metrics.

    Usage::

        from coordinator.monitoring import metrics

        my_counter = metrics.counter("requests_total", "Total requests", ["method"])
        my_counter.labels(method="GET").inc()
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self._registry = registry or REGISTRY
        self._metrics = {}

    def counter(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ) -> Counter:
        """Create or retrieve a Counter metric."""
        if name in self._metrics:
            return self._metrics[name]

        labels = labels or []
        metric = Counter(name, description, labels, registry=self._registry)
        self._metrics[name] = metric
        return metric

    def histogram(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> Histogram:
        """Create or retrieve a Histogram metric."""
        if name in self._metrics:
            return self._metrics[name]

        labels = labels or []
        kwargs = {}
        if buckets:
            kwargs["buckets"] = buckets

        metric = Histogram(
            name, description, labels, registry=self._registry, **kwargs
        )
        self._metrics[name] = metric
        return metric

    def gauge(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ) -> Gauge:
        """Create or retrieve a Gauge metric."""
        if name in self._metrics:
            return self._metrics[name]

        labels = labels or []
        metric = Gauge(name, description, labels, registry=self._registry)
        self._metrics[name] = metric
        return metric


# Module-level singleton used by all coordinator modules
metrics = MetricsManager()
