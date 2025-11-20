# GPU Sizing Calculator

## Purpose
This script is intended to provide preliminary planning and estimation for GPU needs for a given large language
model (LLM) and inference workload characteristics. It helps you determine **how many GPUs** are needed to serve (inferencing) an LLM workload based on:

- **Model architecture parameters** (size, layers, hidden dimension, precision)
- **Workload requirements** (average input/output tokens, requests per second, concurrency)
- **GPU hardware characteristics** (VRAM, memory bandwidth, compute performance in TFLOPs, cost)
- **Real-world performance grounding** from **MLPerf** benchmark reference numbers 

It accounts for two **top-level constraints**:
1. **Memory bound** - GPUs needed to fit model weights + KV cache.
2. **Throughput bound** - GPUs needed to meet tokens-per-second throughput goals.

It also identifies whether the throughput limit is due to **compute bottleneck** or **memory bandwidth bottleneck**.

The output shows GPU needs and how the result is derived through constraints.
---

## How to Use

1. **Create a configuration file** `params.yaml` with parameters for:
   - `model_params`: architecture details of the LLM.
   - `workload_params`: request/workload profile.
   - `gpu_params`: (optional) specify a GPU to evaluate; otherwise, auto-select.
   - `efficiency_factor`: multiplier for real-world performance.
   - `inference_params`: prompt phase (prefill) throughput share.

   Example:
   ```yaml
   model_params:
     PARAMS: 70_000_000_000
     PREC_BITS: 16
     LAYERS: 80
     D_MODEL: 8192
     TOK_LEN: 4096

   workload_params:
     avg_input_tokens: 2000
     avg_output_tokens: 500
     requests_per_sec: 50
     CONCUR: 16
     GPUS_PER_SERVER: 8
     TPS_REQ:          # leave blank to derive automatically

   gpu_params:
     GPU_TYPE:
     VRAM_GPU:
     BW_GPU:
     MIG_SLICES:

   efficiency_factor:
     value: 0.6
   
   inference_params:
     prompt_phase_factor: 0.01 

2. Install dependencies:
    ```bash
    pip install requirements.txt
    ```

3. Run the script:
    ```bash
    python gpu_sizing.py
    -h, --help for help
    -c, --config CONFIG for customized config file
    -g, --gpu-db GPU_DB for customized GPU_DB file
    -d, --debug for debug mode
    -d --debug-file <file> to capture logs into a file
    --mlperf-scale {sqrt,linear,none,adaptive}
    ```

---

## Key Features

- **Two levels of sizing constraints:**
 - **Top level is either Memory bound (VRAM capacity) or throughput bound (tokens/sec).
 - **If it is throughput bound, identify the bottleneck being compute or bandwidth
- **Auto-calculation of TPS_REQ from average input/output tokens and requests/sec if provided.
- **Realistic KV cache sizing using avg_output_tokens instead of maximum TOK_LEN.
- **Cost-aware GPU selection: Sorted by total cost.
- **Auto selection or user provided GPU specified in params.yaml file
- **Selection source indicator: Auto vs user-specified GPU based on the GPU section in the input file.
- **Tokens per second number derived from realistic MLPerf numbers if available, otherwise it is calculated with an effiency factor
- **If fractional GPU (slicing) is required and supported, provide fractional GPU units
- **Debug mode to show results of intermediate steps

---

## Terminology

| Term                        | Meaning                                                                                                  |
|-----------------------------|----------------------------------------------------------------------------------------------------------|
| **PARAMS**                  | Number of trainable model parameters.                                                                    |
| **PREC_BITS**               | Precision in bits (e.g., FP16=16 bits, FP32=32 bits).                                                     |
| **LAYERS**                   | Number of transformer blocks (excludes embedding/output layers).                                        |
| **D_MODEL**                 | Hidden dimension size (per layer) of the model.                                                                      |
| **avg_input_tokens** | Average prompt length in tokens per request. |
| **avg_output_tokens** | Average generated output length in tokens per request. |
| **requests_per_sec** | Number of active incoming requests per second. |
| **CONCUR**                   | Concurrent sequences/users served simultaneously.                                                       |
| **TPS_REQ**                  | Total tokens per second target; combined across all concurrent users.                                   |
| **VRAM_GPU**                 | GPU VRAM capacity (GB).                                                                                 |
| **BW_GPU**                   | GPU memory bandwidth (GB/s).                                                                            |
| **TFLOPS_FP16**              | Peak FP16 tensor-core compute throughput for GPU.                                                       |
| **Memory requirement**       | GPUs needed purely to fit model and KV cache into VRAM.                                                  |
| **Throughput requirement**   | GPUs needed purely to meet TPS target.                                                                  |
| **Throughput bottleneck**    | Whether per-GPU speed is limited by compute or memory bandwidth.                                         |
| **Limiting TPS/unit**        | TPS achieved by one GPU unit given bottleneck, after applying efficiency_factor.                        |
| **TPS (Bandwidth)**          | TPS assuming only bandwidth limits performance.                                                         |
| **TPS (Compute)**            | TPS assuming only compute limits performance.                                                            |
| **Final GPUs used**          | The max of Memory requirement and Throughput requirement.                                               |

---

Sample parameter files are provided for your information under params directory.

Results from this tool are provided as guidance for planning and evaluation purposes only. They are based on general assumptions and may not reflect all real‑world variables. Ensure you validate results before implementation.