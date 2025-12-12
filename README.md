# flash-attention-SWAA
Customized flash attention 2 for SWAA (sliding window attention adaptation)

This is a modified version of [FlashAttention 2.8.3](https://github.com/Dao-AILab/flash-attention), which supports:
1. Keeping attention to the first N tokens in the sequence.
2. Automatically switching from sliding window attention to full attention in the decoding stage (when q_length=1).


# Installation
The ``flash-attention`` folder contains the modified flash attention code, mainly used for inference and training with HuggingFace Transformer.

And the `flash-attention-vllm` folder is the modified flash attention code customized for vLLM, which is slightly different when using Paged-Attention in vLLM.

You can install both by running:
```bash
bash install.sh
```

Or you can install only the customized flash-attention package by running:
```bash
cd flash-attention
MAX_JOBS=4 python setup.py install
```

## Usage Example
<details>
<summary>flash_attn_func</summary>

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
        force_fa_decode=True, # automatically switch to full attention when q_length=1
    )

    end_time = time.time()
    time_taken= end_time - start_time
    print(f"Sliding window: {sliding_window}, Time taken: {time_taken:.4f} seconds")

```
</details>

<details>
<summary>flash_attn_varlen_func</summary>

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
    force_fa_decode=True,  # Automatically switch to full attention when q_length=1
)

print(f"Output shape: {attn_output.shape}")
# Expected output: torch.Size([total_q, 32, 128])
```
</details>

<details>
<summary>Directly using "flash_attn_2_cuda.varlen_fwd" (the underlying function of "flash_attn_varlen_func") with Paged Attention</summary>

```python
import torch
import flash_attn_2_cuda

torch.manual_seed(0)
# parameters
num_heads = 8
head_dim = 64
seqlen_q = [100, 150,300,200]  # the query length of each sequence in the batch
seqlen_k = [1000, 1500,2000,1500] # the key/value length of each sequence in the batch
max_seqlen_q=max(seqlen_q)
max_seqlen_k=max(seqlen_k)
batch_size = len(seqlen_q)
block_size=16
num_blocks=1000

# generate input tensors
total_q = sum(seqlen_q)

q = torch.randn(total_q, num_heads, head_dim, device="cuda", dtype=torch.float16) #shape=(total_q_len, num_heads, head_dim)
k = torch.randn(num_blocks, block_size, num_heads,head_dim, device="cuda", dtype=torch.float16) #shape=(num_blocks, block_size, num_heads, head_dim)
v = torch.randn(num_blocks, block_size, num_heads,head_dim, device="cuda", dtype=torch.float16) #shape=(num_blocks, block_size, num_heads, head_dim)

#generate cu_seqlens_q from seqlen_q
cu_seqlens_q=[0]
for seqlen in seqlen_q:
    cu_seqlens_q.append(cu_seqlens_q[-1]+seqlen)
cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cuda")

# randomly generate block_table. block_id=0 means an empty block
block_table=torch.randint(0, num_blocks, [batch_size, max_seqlen_k//batch_size+1], device="cuda", dtype=torch.int32)

# sequence lengths for k/v
cu_seqlens_k = torch.zeros_like(cu_seqlens_q) # cu_seqlens_k is actually not used so we pass all zeros
seqused_k=torch.tensor(seqlen_k, dtype=torch.int32, device="cuda")

# call the flash_attn_2_cuda.varlen_fwd function, using Paged Attention with block_table
output = flash_attn_2_cuda.varlen_fwd(
    q, k, v,
    None, # out
    cu_seqlens_q,
    cu_seqlens_k,# cu_seqlens_k not used since we use seqused_k, but flash_api.cpp still wants it so we pass all zeros
    seqused_k,# seqused_k decides the actual seqlen_k (i.e. how many KV-blocks will be used) for each sequence
    None, # leftpad_k
    block_table,
    None, #alibi_slopes
    max_seqlen_q, # max_seqlen_q
    max_seqlen_k, # max_seqlen_k
    0.0, # dropout_p
    head_dim**0.5, #softmax_scale
    False, # zero_tensors
    True, # causal
    2000, # window size left
    0, # window size right
    1.0, # softcap
    10, # keep first
    True,  # force_fa_decode
    False,  # return_softmax
    None,
)

print(output[0].shape)  # 输出: torch.Size([total_q_len, num_heads, head_dim])
print("output mean:",output[0].mean().item())
```
</details>