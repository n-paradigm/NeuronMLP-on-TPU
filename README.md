# NeuronMLP-on-TPU

This repository presents the TPU operator migration and optimization results based on the [NeuronMLP](https://github.com/PASAUCMerced/NeuronMM) framework. It provides comprehensive benchmarking tools to evaluate the latency and end-to-end (E2E) acceleration of MLP layers when deployed on TPU hardware, specifically focusing on SVD-compressed model weights.

## 🚀 Features

- **Operator-Level Benchmarking:** Measure the raw execution latency of a single MLP layer on TPU.
- **End-to-End Evaluation:** Assess real-world E2E acceleration ratios using actual SVD-compressed model weights (e.g., LLaMA3-8B).
- **Seamless TPU Migration:** Adapted and optimized from the original NeuronMLP GPU operators to native TPU implementations via JAX/PyTorch XLA.

## 🗂️ Project Structure

```text
├── svdflash_tpu/               # Core TPU operator implementations and benchmarks
│   ├── bench/
│   │   ├── e2e_block.py        # End-to-end MLP block benchmark script          
│   └── speed_up.py                    # Migrated TPU operator kernels as well as the speedup_script
├── results/                    # Directory for benchmark output JSONs         
└── README.md
```

## 📋 Prerequisites

```text
Python >= 3.8
JAX / PyTorch XLA (matching your TPU runtime version)
Access to a TPU runtime (Cloud TPU or local TPU device)
Original NeuronMLP dependencies
```

## 🔧 Installation

Clone the repository and install the required dependencies:
```text
git clone https://github.com/n-paradigm/NeuronMLP-on-TPU.git
cd NeuronMLP-on-TPU
```
## 🏃 Usage

End-to-End Acceleration Test (e2e_block)
Measure the end-to-end speedup ratio of MLP blocks under real, SVD-compressed model weights compared to the baseline.


```bash
python -m svdflash_tpu.bench.e2e_block \
    --model llama3-8b \
    --model-path /path/to/your/llama3-8b_0.9_svd \
    --seq-lens 1024 \
    --warmup 3 \
    --repeats 10 \
    --output results/e2e_8b.json
```
