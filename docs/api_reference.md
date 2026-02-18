# API Reference

## Table of Contents
1. [Overview](#overview)
2. [Base URL](#base-url)
3. [Authentication](#authentication)
4. [Endpoints](#endpoints)
   - [Health Check](#health-check)
   - [List Workers](#list-workers)
   - [List Models](#list-models)
   - [Load Model](#load-model)
   - [Unload Model](#unload-model)
   - [Inference (Completions)](#inference-completions)
   - [Streaming Inference](#streaming-inference)
   - [Batch Inference](#batch-inference)
   - [Get Worker Status](#get-worker-status)
   - [Get Model Status](#get-model-status)
   - [Metrics](#metrics)
5. [Data Types](#data-types)
6. [Error Handling](#error-handling)
7. [Rate Limiting](#rate-limiting)
8. [Examples](#examples)
9. [SDKs](#sdks)

---

## Overview

The AI Cluster API provides a RESTful interface to interact with the distributed inference cluster. All endpoints return JSON responses and support both synchronous and asynchronous operations.

**Base URL**: `http://<coordinator-host>:8000/v1`

**API Version**: v1

**Content Type**: `application/json`

---

## Base URL

```
http://localhost:8000/v1
```

For production deployments:
```
https://ai-cluster.example.com/v1
```

---

## Authentication

### API Key Authentication

Include your API key in the `Authorization` header:

```http
Authorization: Bearer sk-your-api-key-here
```

### Example

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Authorization: Bearer sk-1234567890abcdef" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-7b",
    "prompt": "Hello, how are you?",
    "max_tokens": 100
  }'
```

### Rate Limits

| Tier | Requests per minute | Tokens per minute |
|------|--------------------|-------------------|
| Free | 60 | 10,000 |
| Pro | 600 | 100,000 |
| Enterprise | Custom | Custom |

---

## Endpoints

### Health Check

Check if the coordinator is healthy.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "workers": 4,
  "version": "0.1.0",
  "uptime_seconds": 3600
}
```

**Status Codes**:
- `200`: Healthy
- `503`: Unhealthy (no workers available)

---

### List Workers

Get information about all connected workers.

**Endpoint**: `GET /v1/workers`

**Response**:
```json
{
  "workers": [
    {
      "id": "amd-gpu-0",
      "address": "192.168.1.100:50051",
      "state": "healthy",
      "gpus": [
        {
          "id": 0,
          "name": "AMD Radeon 9060 XT",
          "memory_gb": 16.0,
          "available_gb": 12.5,
          "utilization": 45.2,
          "temperature": 65.3
        }
      ],
      "loaded_models": ["deepseek-7b"],
      "active_requests": 2,
      "total_requests": 1500,
      "avg_latency_ms": 125.4
    }
  ],
  "total_workers": 1,
  "total_gpus": 1,
  "total_memory_gb": 16.0,
  "available_memory_gb": 12.5
}
```

---

### List Models

Get information about available models.

**Endpoint**: `GET /v1/models`

**Response**:
```json
{
  "models": [
    {
      "name": "deepseek-7b",
      "family": "deepseek",
      "parameters": "7B",
      "quantization": ["fp16", "int8", "int4"],
      "min_memory_gb": 16,
      "loaded_on": [
        {
          "worker_id": "amd-gpu-0",
          "gpus": [0]
        }
      ],
      "description": "DeepSeek 7B Base Model with MoE architecture"
    },
    {
      "name": "llama3-8b",
      "family": "llama",
      "parameters": "8B",
      "quantization": ["fp16", "int8"],
      "min_memory_gb": 16,
      "loaded_on": [],
      "description": "Meta Llama 3 8B Instruct"
    }
  ]
}
```

---

### Load Model

Load a model onto available workers.

**Endpoint**: `POST /v1/models/load`

**Request Body**:
```json
{
  "model_name": "deepseek-7b",
  "quantization": "fp16",
  "parallelism": "auto",
  "gpu_ids": [0, 1],
  "worker_ids": ["amd-gpu-0"],
  "timeout": 300
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| model_name | string | Yes | Name of the model to load |
| quantization | string | No | Quantization type: fp16, int8, int4 (default: fp16) |
| parallelism | string | No | Parallelism strategy: auto, pipeline, tensor, data, expert |
| gpu_ids | array | No | Specific GPU IDs to use (empty = auto-select) |
| worker_ids | array | No | Specific worker IDs to use |
| timeout | integer | No | Load timeout in seconds (default: 300) |

**Response**:
```json
{
  "status": "loaded",
  "model_name": "deepseek-7b",
  "workers": [
    {
      "worker_id": "amd-gpu-0",
      "gpu_ids": [0],
      "memory_used_gb": 8.2,
      "load_time_ms": 15420
    }
  ],
  "total_memory_gb": 8.2
}
```

**Status Codes**:
- `200`: Model loaded successfully
- `202`: Loading in progress (async)
- `400`: Invalid request
- `503`: No workers available

---

### Unload Model

Unload a model from workers.

**Endpoint**: `DELETE /v1/models/{model_name}`

**Parameters**:
- `model_name`: Name of the model to unload

**Query Parameters**:
- `worker_id`: Specific worker to unload from (optional)
- `force`: Force unload even if busy (default: false)

**Response**:
```json
{
  "status": "unloaded",
  "model_name": "deepseek-7b",
  "workers": ["amd-gpu-0"],
  "memory_freed_gb": 8.2
}
```

---

### Inference (Completions)

Generate text completions.

**Endpoint**: `POST /v1/completions`

**Request Body**:
```json
{
  "model": "llama-3-8b-instruct",
  "prompt": "Explain quantum computing in simple terms",
  "max_tokens": 200,
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 40,
  "stop": ["\n", "Human:", "Assistant:"],
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stream": false,
  "timeout": 60,
  "seed": 42,
  "echo": false,
  "logprobs": false,
  "suffix": null
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| model | string | required | Model to use for completion |
| prompt | string | required | Input text prompt |
| max_tokens | integer | 512 | Maximum tokens to generate |
| temperature | float | 0.7 | Sampling temperature (0-2) |
| top_p | float | 0.95 | Nucleus sampling parameter |
| top_k | integer | 40 | Top-k sampling parameter |
| stop | array | [] | Stop sequences |
| presence_penalty | float | 0.0 | Penalize new tokens |
| frequency_penalty | float | 0.0 | Penalize frequent tokens |
| stream | boolean | false | Stream response incrementally |
| timeout | integer | 60 | Request timeout in seconds |
| seed | integer | null | Random seed for reproducibility |
| echo | boolean | false | Echo the prompt in response |
| logprobs | boolean | false | Return log probabilities |
| suffix | string | null | Text to append after generation |

**Response (Non-streaming)**:
```json
{
  "id": "cmpl-123456",
  "object": "text_completion",
  "created": 1677858242,
  "model": "deepseek-7b",
  "choices": [
    {
      "text": "\n\nQuantum computing is a type of computing that uses quantum-mechanical phenomena...",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop",
      "tokens_generated": 156
    }
  ],
  "usage": {
    "prompt_tokens": 7,
    "completion_tokens": 156,
    "total_tokens": 163
  },
  "worker_id": "amd-gpu-0",
  "processing_time_ms": 1245
}
```

---

### Streaming Inference

Stream tokens as they're generated.

**Endpoint**: `POST /v1/completions` with `stream: true`

**Request**:
```json
{
  "model": "deepseek-7b",
  "prompt": "Tell me a story",
  "max_tokens": 100,
  "stream": true
}
```

**Response (Server-Sent Events)**:
```
event: completion
data: {"id":"cmpl-123456","object":"completion.chunk","created":1677858242,"choices":[{"index":0,"text":"Once","finish_reason":null,"tokens_generated":1}]}

event: completion
data: {"id":"cmpl-123456","object":"completion.chunk","created":1677858242,"choices":[{"index":0,"text":" upon","finish_reason":null,"tokens_generated":2}]}

event: completion
data: {"id":"cmpl-123456","object":"completion.chunk","created":1677858242,"choices":[{"index":0,"text":" a","finish_reason":null,"tokens_generated":3}]}

event: completion
data: {"id":"cmpl-123456","object":"completion.chunk","created":1677858242,"choices":[{"index":0,"text":" time","finish_reason":null,"tokens_generated":4}]}

event: done
data: {"id":"cmpl-123456","object":"completion","created":1677858242,"choices":[{"index":0,"text":"","finish_reason":"stop","tokens_generated":100}],"usage":{"prompt_tokens":4,"completion_tokens":100,"total_tokens":104}}
```

**Client Example (Python)**:
```python
import httpx
import json

async def stream_completion():
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/v1/completions",
            json={
                "model": "deepseek-7b",
                "prompt": "Tell me a story",
                "stream": True
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if "choices" in data:
                        text = data["choices"][0]["text"]
                        print(text, end="", flush=True)
```

---

### Batch Inference

Process multiple prompts in a single request.

**Endpoint**: `POST /v1/completions/batch`

**Request Body**:
```json
{
  "model": "deepseek-7b",
  "prompts": [
    "Explain quantum computing",
    "What is machine learning?",
    "Write a poem about AI"
  ],
  "max_tokens": 100,
  "temperature": 0.7
}
```

**Response**:
```json
{
  "id": "batch-123456",
  "object": "batch",
  "created": 1677858242,
  "model": "deepseek-7b",
  "completions": [
    {
      "index": 0,
      "text": "\n\nQuantum computing is...",
      "tokens_generated": 85,
      "finish_reason": "stop"
    },
    {
      "index": 1,
      "text": "\n\nMachine learning is...",
      "tokens_generated": 92,
      "finish_reason": "stop"
    },
    {
      "index": 2,
      "text": "\n\nRoses are red,\nViolets are blue,\nAI is awesome...",
      "tokens_generated": 78,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 21,
    "completion_tokens": 255,
    "total_tokens": 276
  },
  "processing_time_ms": 3450
}
```

---

### Get Worker Status

Get detailed status of a specific worker.

**Endpoint**: `GET /v1/workers/{worker_id}`

**Response**:
```json
{
  "id": "amd-gpu-0",
  "address": "192.168.1.100:50051",
  "state": "healthy",
  "version": "0.1.0",
  "uptime_seconds": 86400,
  "gpus": [
    {
      "id": 0,
      "name": "AMD Radeon 9060 XT",
      "memory_gb": 16.0,
      "used_gb": 8.2,
      "free_gb": 7.8,
      "utilization": 45.2,
      "temperature": 65.3,
      "power_watts": 185,
      "capabilities": ["fp16", "int8"]
    }
  ],
  "loaded_models": [
    {
      "name": "deepseek-7b",
      "memory_gb": 8.2,
      "quantization": "fp16",
      "parallelism": "single",
      "loaded_at": "2024-01-15T10:30:00Z",
      "inference_count": 1250
    }
  ],
  "performance": {
    "active_requests": 2,
    "queued_requests": 0,
    "avg_latency_ms": 125.4,
    "p95_latency_ms": 245.8,
    "p99_latency_ms": 312.3,
    "requests_per_second": 4.5,
    "tokens_per_second": 112.5
  }
}
```

---

### Get Model Status

Get detailed status of a specific model across all workers.

**Endpoint**: `GET /v1/models/{model_name}`

**Response**:
```json
{
  "name": "deepseek-7b",
  "family": "deepseek",
  "parameters": "7B",
  "config": {
    "num_layers": 30,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "vocab_size": 32256,
    "max_seq_len": 4096
  },
  "instances": [
    {
      "worker_id": "amd-gpu-0",
      "gpu_ids": [0],
      "quantization": "fp16",
      "parallelism": "single",
      "memory_gb": 8.2,
      "loaded_at": "2024-01-15T10:30:00Z",
      "inference_count": 1250,
      "avg_latency_ms": 125.4
    },
    {
      "worker_id": "nvidia-gpu-0",
      "gpu_ids": [0],
      "quantization": "int8",
      "parallelism": "single",
      "memory_gb": 4.1,
      "loaded_at": "2024-01-15T11:15:00Z",
      "inference_count": 850,
      "avg_latency_ms": 142.1
    }
  ],
  "total_inferences": 2100,
  "avg_latency_ms": 132.5
}
```

---

### Metrics

Get Prometheus-formatted metrics.

**Endpoint**: `GET /metrics`

**Response** (Prometheus text format):
```
# HELP coordinator_requests_total Total requests processed
# TYPE coordinator_requests_total counter
coordinator_requests_total{model="deepseek-7b",status="success"} 1250
coordinator_requests_total{model="llama3-8b",status="success"} 850

# HELP coordinator_request_duration_seconds Request duration in seconds
# TYPE coordinator_request_duration_seconds histogram
coordinator_request_duration_seconds_bucket{model="deepseek-7b",le="0.1"} 120
coordinator_request_duration_seconds_bucket{model="deepseek-7b",le="0.5"} 980
coordinator_request_duration_seconds_bucket{model="deepseek-7b",le="1.0"} 1180
coordinator_request_duration_seconds_bucket{model="deepseek-7b",le="5.0"} 1245

# HELP worker_gpu_utilization_percent GPU utilization percentage
# TYPE worker_gpu_utilization_percent gauge
worker_gpu_utilization_percent{worker="amd-gpu-0",gpu="0"} 45.2
```

---

## Data Types

### GPUInfo

| Field | Type | Description |
|-------|------|-------------|
| id | integer | GPU index |
| name | string | GPU model name |
| memory_gb | float | Total VRAM in GB |
| available_gb | float | Free VRAM in GB |
| used_gb | float | Used VRAM in GB |
| utilization | float | GPU utilization (0-100) |
| temperature | float | Temperature in Celsius |
| power_watts | integer | Power usage in watts |
| capabilities | array | GPU capabilities |

### ModelInfo

| Field | Type | Description |
|-------|------|-------------|
| name | string | Model identifier |
| family | string | Model family (deepseek, llama, etc.) |
| parameters | string | Parameter count (e.g., "7B") |
| quantization | array | Supported quantization types |
| min_memory_gb | float | Minimum VRAM required |
| description | string | Model description |
| loaded_on | array | Workers where model is loaded |

### CompletionChoice

| Field | Type | Description |
|-------|------|-------------|
| text | string | Generated text |
| index | integer | Choice index |
| logprobs | object | Log probabilities (if requested) |
| finish_reason | string | Why generation stopped (stop, length, timeout, error) |
| tokens_generated | integer | Number of tokens generated |

### Usage

| Field | Type | Description |
|-------|------|-------------|
| prompt_tokens | integer | Tokens in the prompt |
| completion_tokens | integer | Tokens generated |
| total_tokens | integer | Total tokens processed |

---

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Rate limit exceeded. Please try again in 30 seconds.",
    "details": {
      "limit": 60,
      "remaining": 0,
      "reset_after": 30
    },
    "request_id": "req-123456"
  }
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid or missing API key |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource not found |
| 408 | Request Timeout - Request took too long |
| 409 | Conflict - Model already loading |
| 422 | Unprocessable Entity - Validation error |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 502 | Bad Gateway - Worker unavailable |
| 503 | Service Unavailable - No workers available |
| 504 | Gateway Timeout - Worker timed out |

### Error Codes

| Code | Description |
|------|-------------|
| model_not_found | Requested model not available |
| model_not_loaded | Model not loaded on any worker |
| no_workers_available | No healthy workers available |
| insufficient_memory | Not enough GPU memory |
| rate_limit_exceeded | API rate limit exceeded |
| invalid_parameter | Invalid request parameter |
| authentication_failed | Invalid API key |
| request_timeout | Request timed out |
| worker_unavailable | Worker is offline |
| model_load_failed | Failed to load model |
| inference_failed | Inference execution failed |

---

## Rate Limiting

### Headers

| Header | Description |
|--------|-------------|
| X-RateLimit-Limit | Requests per minute limit |
| X-RateLimit-Remaining | Remaining requests in current window |
| X-RateLimit-Reset | Seconds until rate limit resets |

### Example

```http
HTTP/1.1 200 OK
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 30
```

---

## Examples

### Python Example

```python
import httpx
import asyncio

class AIClusterClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.client = httpx.AsyncClient(headers=self.headers)
    
    async def completion(self, prompt: str, model: str = "deepseek-7b", **kwargs):
        response = await self.client.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                **kwargs
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def stream_completion(self, prompt: str, model: str = "deepseek-7b", **kwargs):
        async with self.client.stream(
            "POST",
            f"{self.base_url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "stream": True,
                **kwargs
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield line[6:]
    
    async def list_models(self):
        response = await self.client.get(f"{self.base_url}/v1/models")
        return response.json()
    
    async def close(self):
        await self.client.aclose()

# Usage
async def main():
    client = AIClusterClient(
        base_url="http://localhost:8000/v1",
        api_key="sk-1234567890abcdef"
    )
    
    # Non-streaming
    result = await client.completion(
        prompt="Explain quantum computing",
        max_tokens=100,
        temperature=0.7
    )
    print(result["choices"][0]["text"])
    
    # Streaming
    async for chunk in client.stream_completion("Tell me a story"):
        data = json.loads(chunk)
        print(data["choices"][0]["text"], end="", flush=True)
    
    await client.close()

asyncio.run(main())
```

### JavaScript/Node.js Example

```javascript
class AIClusterClient {
  constructor(baseUrl, apiKey) {
    this.baseUrl = baseUrl;
    this.headers = {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    };
  }

  async completion(prompt, options = {}) {
    const response = await fetch(`${this.baseUrl}/v1/completions`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({
        model: options.model || 'deepseek-7b',
        prompt,
        max_tokens: options.maxTokens || 100,
        temperature: options.temperature || 0.7,
        ...options
      })
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error.message);
    }
    
    return response.json();
  }

  async *streamCompletion(prompt, options = {}) {
    const response = await fetch(`${this.baseUrl}/v1/completions`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({
        model: options.model || 'deepseek-7b',
        prompt,
        max_tokens: options.maxTokens || 100,
        temperature: options.temperature || 0.7,
        stream: true,
        ...options
      })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data !== '[DONE]') {
            yield JSON.parse(data);
          }
        }
      }
    }
  }

  async listModels() {
    const response = await fetch(`${this.baseUrl}/v1/models`, {
      headers: this.headers
    });
    return response.json();
  }
}

// Usage
const client = new AIClusterClient(
  'http://localhost:8000/v1',
  'sk-1234567890abcdef'
);

// Non-streaming
client.completion('Explain quantum computing')
  .then(result => console.log(result.choices[0].text));

// Streaming
(async () => {
  for await (const chunk of client.streamCompletion('Tell me a story')) {
    process.stdout.write(chunk.choices[0].text);
  }
})();
```

### curl Examples

```bash
# Health check
curl http://localhost:8000/health

# List workers
curl -H "Authorization: Bearer sk-1234567890abcdef" \
  http://localhost:8000/v1/workers

# List models
curl -H "Authorization: Bearer sk-1234567890abcdef" \
  http://localhost:8000/v1/models

# Load model
curl -X POST http://localhost:8000/v1/models/load \
  -H "Authorization: Bearer sk-1234567890abcdef" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "deepseek-7b"}'

# Inference
curl -X POST http://localhost:8000/v1/completions \
  -H "Authorization: Bearer sk-1234567890abcdef" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-7b",
    "prompt": "What is artificial intelligence?",
    "max_tokens": 100
  }'

# Streaming inference
curl -X POST http://localhost:8000/v1/completions \
  -H "Authorization: Bearer sk-1234567890abcdef" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-7b",
    "prompt": "Tell me a story",
    "max_tokens": 200,
    "stream": true
  }'

# Batch inference
curl -X POST http://localhost:8000/v1/completions/batch \
  -H "Authorization: Bearer sk-1234567890abcdef" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-7b",
    "prompts": [
      "What is AI?",
      "Explain quantum computing"
    ],
    "max_tokens": 50
  }'

# Get worker status
curl -H "Authorization: Bearer sk-1234567890abcdef" \
  http://localhost:8000/v1/workers/amd-gpu-0

# Get model status
curl -H "Authorization: Bearer sk-1234567890abcdef" \
  http://localhost:8000/v1/models/deepseek-7b

# Metrics
curl http://localhost:8000/metrics
```

---

## SDKs

### Python SDK

```bash
pip install ai-cluster-client
```

```python
from ai_cluster import Client

client = Client(
    base_url="http://localhost:8000",
    api_key="sk-1234567890abcdef"
)

# Simple completion
response = client.complete(
    model="deepseek-7b",
    prompt="Hello!",
    max_tokens=100
)
print(response.text)

# Streaming
for chunk in client.stream_complete(
    model="deepseek-7b",
    prompt="Tell me a story"
):
    print(chunk.text, end="")

# Async
async for chunk in client.async_stream_complete(
    model="deepseek-7b",
    prompt="Tell me a story"
):
    print(chunk.text, end="")
```

### JavaScript SDK

```bash
npm install ai-cluster-client
```

```javascript
import { AIClusterClient } from 'ai-cluster-client';

const client = new AIClusterClient({
  baseUrl: 'http://localhost:8000',
  apiKey: 'sk-1234567890abcdef'
});

// Simple completion
const response = await client.complete({
  model: 'deepseek-7b',
  prompt: 'Hello!',
  maxTokens: 100
});
console.log(response.text);

// Streaming
const stream = client.streamComplete({
  model: 'deepseek-7b',
  prompt: 'Tell me a story'
});
for await (const chunk of stream) {
  process.stdout.write(chunk.text);
}
```

### Go SDK

```go
package main

import (
    "context"
    "fmt"
    "github.com/ai-cluster/client-go"
)

func main() {
    client := aicluster.NewClient("http://localhost:8000", "sk-1234567890abcdef")
    
    resp, err := client.Complete(context.Background(), &aicluster.CompletionRequest{
        Model:     "deepseek-7b",
        Prompt:    "Hello!",
        MaxTokens: 100,
    })
    if err != nil {
        panic(err)
    }
    fmt.Println(resp.Text)
}
```

---

## WebSocket Support

For real-time applications, the API also supports WebSocket connections.

**Endpoint**: `ws://localhost:8000/v1/ws`

### Example

```javascript
const ws = new WebSocket('ws://localhost:8000/v1/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'completion',
    model: 'deepseek-7b',
    prompt: 'Tell me a story',
    max_tokens: 100
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'chunk') {
    console.log(data.text);
  } else if (data.type === 'done') {
    console.log('Complete!');
    ws.close();
  }
};
```

---

## OpenAPI Specification

The API is fully documented using OpenAPI 3.0. You can access the specification at:

```
http://localhost:8000/openapi.json
```

Or view the interactive documentation at:

```
http://localhost:8000/docs
http://localhost:8000/redoc
```

---

## Changelog

### v0.1.0 (2024-01-15)
- Initial release
- Basic completion API
- Model loading/unloading
- Worker management
- Streaming support
- Prometheus metrics

### v0.2.0 (Planned)
- Batch inference
- Embeddings API
- Fine-tuning support
- Multi-modal models
- A/B testing

---

For more information, see:
- [Architecture Guide](architecture.md)
- [Deployment Guide](deployment.md)
- [Configuration Guide](configuration.md)
- [Troubleshooting](troubleshooting.md)