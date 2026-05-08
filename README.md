# MiniMax-M2.5 on Azure ML

Deploy [MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) (230B MoE, 10B activated) as an Azure ML Managed Online Endpoint using vLLM.

## Two Deployment Variants

| | NC A100 | ND A100 |
|---|---|---|
| **Notebook** | `deploy_nca100-minimax25.ipynb` | `deploy_nda100-minimax25.ipynb` |
| **Config** | `config_nca100-minimax25/` | `config_nda100-minimax25/` |
| **VM** | `Standard_NC96ads_A100_v4` | `Standard_ND96amsr_A100_v4` |
| **GPUs** | 4× A100 80GB PCIe | 8× A100 80GB SXM NVLink |
| **Strategy** | TP4 | DP8+EP |
| **Workarounds** | NCCL P2P disable, eager mode, Triton FP8 disable | Triton FP8 disable only |

> **Region:** Deploy in any Azure region where the target VM SKU is available. Update `AZURE_WORKSPACE_NAME` and `AZURE_RESOURCE_GROUP` in `.env` to match your workspace.

## Setup

1. Copy `.env.example` → `.env` and fill in your Azure credentials
2. `az login --tenant <your-tenant-id>`
3. Open the notebook for your target VM and run cells top-to-bottom

## Files

```
.env.example                    # Template for Azure credentials
config_nca100-minimax25/        # NC A100 PCIe config (Dockerfile, YAML)
config_nda100-minimax25/        # ND A100 NVLink config (Dockerfile, YAML)
deploy_nca100-minimax25.ipynb   # Deployment notebook — NC A100
deploy_nda100-minimax25.ipynb   # Deployment notebook — ND A100
MIGRATION_ND_A100.md            # Migration guide NC → ND
```
