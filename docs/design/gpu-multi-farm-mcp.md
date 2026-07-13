# GPU Multi-Farm MCP

MCP server for querying and launching training GPUs across **Vast.ai**, **RunPod**, and **Lambda Labs**.

## Tools

| Tool | Purpose |
| --- | --- |
| `list_available_gpus` | Cross-farm availability + pricing |
| `launch_training_pod` | Launch a training-oriented pod |
| `project_training_cost` | Hours × price × Cactus overhead; recommend cheapest |

## Modes

Controlled by `GPU_MULTI_FARM_MODE`:

- `auto` (default): live client when the farm API key is set, otherwise mock
- `live`: always call provider APIs (missing key → `missing_api_key` error for that farm)
- `mock`: deterministic fixture offers (CI / offline)

Mock launches are **refused** unless `config.allow_mock_launch=true`.

## Auth

| Farm | Env var | API |
| --- | --- | --- |
| Vast.ai | `VAST_API_KEY` | `https://console.vast.ai/api/v0` |
| RunPod | `RUNPOD_API_KEY` | GraphQL `https://api.runpod.io/graphql` |
| Lambda | `LAMBDA_API_KEY` | `https://cloud.lambda.ai/api/v1` |

See [`.env.example`](../../.env.example).

## Cactus overhead

`CACTUS_OVERHEAD` (default `1.08`) scales projected compute cost to account for runtime / KV / orchestration overhead. It is a placeholder multiplier, not a Tensorfeed price feed.

## Cursor MCP config

```json
{
  "mcpServers": {
    "gpu-multi-farm": {
      "command": "python",
      "args": ["-m", "scripts.multi_farm_mcp"],
      "cwd": "/absolute/path/to/slm-training",
      "env": {
        "GPU_MULTI_FARM_MODE": "auto"
      }
    }
  }
}
```

Install deps first: `pip install -e ".[mcp]"`.

## Non-goals

- Auto-running TwoTower training on the pod
- Destroy/stop tools (future)
- Paid spot-price data feeds
