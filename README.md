# flash-attention-SWAA
Customized flash attention 2 for SWAA (sliding window attention adaptation)

This is a modified version of [FlashAttention 2.8.3](https://github.com/Dao-AILab/flash-attention), which supports:
1. Keeping attention to the first N tokens in the sequence.
2. Automatically switching from sliding window attention to full attention in the decoding stage (when q_length=1).


# Installation
The ``flash-attention`` folder contains the modified flash attention code. You can install it by running:
```bash
bash install.sh
```

## Usage Example
flash_attn_func
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import time
from flash_attn import flash_attn_func

# Prepare some random data
query_length = 10000
query_states = torch.randn(1, query_length, 32, 128, dtype=torch.bfloat16, device="cuda") 
key_states = torch.randn(1, query_length, 4, 128, dtype=torch.bfloat16, device="cuda") 
value_states = torch.randn(1, query_length, 4, 128, dtype=torch.bfloat16, device="cuda") 

# Try different sliding window sizes. -1 means full attention
for sliding_window in [-1,100,1000,4000,8000]:

    start_time = time.time()
    # flash attention forward
    attn_output = flash_attn_func(
        query_states,
        key_states,
        value_states,
        causal=True,
        window_size=(sliding_window, sliding_window), 
        keep_first=4, # for example, we keep the attention to the first 4 tokens
        auto_prefill_slide=True, # automatically switch to full attention when q_length=1
    )

    end_time = time.time()
    time_taken= end_time - start_time
    print(f"Sliding window: {sliding_window}, Time taken: {time_taken:.4f} seconds")

```

flash_attn_varlen_func
```python
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from flash_attn import flash_attn_varlen_func

# Example data for a batch of 2 sequences
# Sequence 1 length: 2000, Sequence 2 length: 3000
seqlens = [1, 2, 2000, 3000]

# Cumulative sequence lengths
# Should be [0, 2000, 5000]
cu_seqlens_q = torch.tensor([0] + list(torch.cumsum(torch.tensor(seqlens), dim=0)), dtype=torch.int32, device="cuda")
cu_seqlens_k = cu_seqlens_q.clone()

total_q = cu_seqlens_q[-1].item()  # Total tokens in batch (5000)
total_k = cu_seqlens_k[-1].item()  # Total tokens in batch (5000)

max_seqlen_q = max(seqlens)  # 3000
max_seqlen_k = max(seqlens)  # 3000

# Model dimensions (GQA example)
nheads = 32
nheads_k = 4
headdim = 128

# Prepare random data
q = torch.randn(total_q, nheads, headdim, dtype=torch.bfloat16, device="cuda")
k = torch.randn(total_k, nheads_k, headdim, dtype=torch.bfloat16, device="cuda")
v = torch.randn(total_k, nheads_k, headdim, dtype=torch.bfloat16, device="cuda")

# Set a sliding window size
sliding_window = 4000

# Flash attention varlen forward
attn_output = flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    causal=True,
    window_size=(sliding_window, sliding_window),
    keep_first=4,  # Keep attention to the first 4 tokens
    auto_prefill_slide=True,  # Automatically switch to full attention when q_length=1
)

print(f"Output shape: {attn_output.shape}")
# Expected output: torch.Size([total_q, 32, 128])

```