import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as sp
import numpy as np

spm = sp.SentencePieceProcessor()
spm.load("/kaggle/input/spm2pro/pytorch/default/1/spm.model")


encode_text = lambda x: spm.encode(x, out_type=int)
decode_text = lambda x: spm.decode(x)

B = 8
T = 1024
d_model = 768
n_layer = 12
n_head = 12  # 768 is divisible by 12 (head_dim=64)
n_kv_heads = 4  # 12 is divisible by 4 (Grouped Query Attention)
dropout = 0.2  # Lower dropout slightly for larger models/more data
# Ensure this matches or exceeds T
context_length = T

vocab_size = spm.vocab_size()
print(vocab_size)

device = "cuda" if torch.cuda.is_available() else "cpu"

class Dataloader:
    def __init__(self,B,T):
        self.B=B
        self.T=T
        all_tokens = np.memmap(
        "/kaggle/input/nepali-tokens2/Nepali_tokens.bin", dtype=np.uint16, mode="r"
       )

        self.n = len(all_tokens)
        split = int(0.05 * self.n)

        train_tokens= all_tokens[split:]
        val_tokens = all_tokens[:split]       
        print(f"Total tokens:{self.n}")
        print(f"train_tokens:{len(train_tokens)}")
        print(f"val_tokens:{len(val_tokens)}")
        train_tokens = train_tokens.astype(np.int64)
        val_tokens = val_tokens.astype(np.int64)
        self.train_tokens=torch.tensor(train_tokens,dtype=torch.long)
        self.val_tokens=torch.tensor(val_tokens,dtype=torch.long)
        self.trainPosition=0
        self.valPosition=0


    def get_batch(self,split):
        if split =="train":
            data=self.train_tokens
            position=self.trainPosition
        else:
            data=self.val_tokens
            position=self.valPosition
            
        num=len(data) 
        B,T=self.B,self.T
        buf=data[position:position+B*T+1]
        x=buf[:-1].view(B,T)
        y=buf[1:].view(B,T)
        position+=B*T
        if position + (B*T)+1 >num:
            position=0
        
        #We have changed position above in the line position+=B*T but it only changes the local varaible and not the actual train or val position
        if split =="train":
            self.trainPosition=position
        else:
            self.valPosition=position 
       
        return x,y            
        

# def get_batch(split, B, T):
#     tokens = train_tokens if split == "train" else val_tokens
#     idx = torch.randint(0, len(tokens) - T - 1, (B,))
#     x = torch.stack([tokens[i : i + T] for i in idx])
#     y = torch.stack([tokens[i + 1 : i + T + 1] for i in idx])
#     return x, y


dataloader=Dataloader(B,T)

# function decorator
@torch.no_grad()
def estimate_loss(B):
    model.eval()
    out = {"train": 0, "val": 0}
    for split in ["train", "val"]:
        losses = []
        num_batches = 50
        for i in range(num_batches):
            x, y = dataloader.get_batch(split)
            x, y = x.to(device=device), y.to(device=device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


class RoPE(nn.Module):
    # Applies a rotation matrix to Q and K based on token position
    def __init__(self, head_dim: int, context_length: int, base: int = 10000):
        super().__init__()
        self.head_dim = head_dim
        self.context_length = context_length
        self.base = base

        # calculate the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute sin and cos terms
        t = torch.arange(context_length, dtype=torch.float32)
        # self.register_buffer("t",torch.arange(context_length, dtype=torch.float32))
        freqs = torch.outer(t, self.inv_freq)

        # emb
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:

        # Rotate the second half of the tensor by 180

        # x shape:(B,n_head,T,head_dim), (B,n_kv_head,T,head_dim)

        x_half = (
            x.contiguous()
            .reshape(*x.shape[:-1], 2, -1)
            .transpose(-2, -1)
            .reshape(x.shape)
        )

        x1, x2 = x_half.chunk(2, dim=-1)

        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        # x shape: (B, n_head, T, head_dim)
        B_curr = x.size(0)
        T_curr = x.size(2)

        # Cast to float32 for RoPE math stability
        x_orig = x.to(torch.float32)
        x_rot = self._rotate_half(x_orig)

        # Use slicing for the current sequence length T
        # Slicing creates a 'view', which is nearly free (zero memory copy)
        cos = self.cos_cache[pos, :]  # (T, D)
        sin = self.sin_cache[pos, :]  # (T, D)

        # Reshape to (1, 1, T, D) so it broadcasts across B and n_head automatically
        cos = cos.unsqueeze(0).unsqueeze(0) # Becomes (1, 1, T, D)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Apply the rotation
        return (x_orig * cos + x_rot * sin).to(x.dtype)


class FeedForward(nn.Module):
    def __init__(self, d_model):
        #We will use Swiglu
        super().__init__()
        self.fc_gate_value = nn.Linear(d_model, 8*d_model)
        self.fc_2 = nn.Linear( 4*d_model, d_model)

    def forward(self, x):
        #1.project to a large dimension
        combined=self.fc_gate_value(x)
        
        # 2. Split the result into two halves: gate and value
        gate,value=combined.chunk(chunks=2,dim=-1)
        
        #3. Apply SwiGLU: SiLU(gate) * value
        x=F.silu(gate) * value
        
        # 4. Project back down
        x = self.fc_2(x)
        return x


class KVCache:
    def __init__(self):
        self.key = None
        self.value = None

    def add(self, key_new, value_new):
        # Shape of both key_new, value_new is (B,n_kv_heads,T,head_dim)

        if self.key is None:
            # Remove a tensor from the autograd computation graph.
            self.key = key_new.detach()
            self.value = value_new.detach()
        else:
            self.key = torch.cat((self.key, key_new), dim=2)
            self.value = torch.cat((self.value, value_new), dim=2)

    def get(self):
        return self.key, self.value


# n_kv_heads is basically number of groups
# n_rep is the group size(number of K or V repeated)
# if n_heads=16, and n_kv_heads(n_group)=4 then n_rep=4
class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, n_head, context_length, droupout, n_kv_heads):
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        assert n_head % n_kv_heads == 0, "n_head must be divisible by n_kv_heads"
        super().__init__()

        self.head_dim = d_model // n_head
        self.d_model = d_model
        self.n_head = n_head
        self.n_kv_heads = n_kv_heads
        # Query has full heads
        self.q_proj = nn.Linear(d_model, self.head_dim * n_head)
        # But Key and Value have fewer heads (n_kv_heads)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        # Number of times K V must be repeated to match the size of Query
        self.n_repeat = n_head // n_kv_heads
        self.droupout = nn.Dropout(droupout)
        self.out_proj = nn.Linear(d_model, d_model)
        # RoPE
        self.rope = RoPE(self.head_dim, context_length=context_length)

    def _repeat_kv(self, n_repeat: int, KV: torch.Tensor) -> torch.Tensor:
        """
        This function repeats the Key/Value heads to match the number of Query heads.
        Input shape:  (batch, n_kv_heads, seq_len, head_dim)
        Output shape: (batch, n_kv_heads, seq_len,   head_dim)
        """
        B, n_kv_heads, T, head_dim = KV.shape
        if n_kv_heads == 1:
            return KV

        # We expand the tensor to duplicate the data n_rep times we want=> n_kv_heads -> n_heads

        # (batch, seq_len, n_kv_heads, n_rep, head_dim)
        KV = KV.repeat_interleave(n_repeat, dim=1)
        return KV

    def forward(self, x: torch.Tensor, pos: torch.Tensor, kv_cache: KVCache = None):
        """
        x: Input tensor of shape (batch_size, seq_len, d_model)
        kv_cache: A tuple (past_key, past_value) containing cached tokens
        """
        if self.training:
            kv_cache = None

        B, T, C = x.shape
        # 1.Project Q, K and V
        Q = self.q_proj(x)  # [B,T,head_dim*n_head]
        K = self.k_proj(x)  # [B,T,head_dim*n_kv_heads]
        V = self.v_proj(x)  # [B,T,head_dim*n_kv_heads]

        # 2.Reshape to separate heads
        # Q:(B,n_head,T,head_dim)
        Q = Q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # K:(B,n_kv_heads,T,head_dim)
        K = K.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # V:(B,n_kv_heads,T,head_dim)
        V = V.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        Q = self.rope(Q, pos)
        K = self.rope(K, pos)

        # 3. KV Cache Management
        if kv_cache is not None:
            kv_cache.add(K, V)
            K, V = kv_cache.get()

        # Repeat for the GQA key and value
        K_expanded = self._repeat_kv(self.n_repeat, K)
        # (B,n_kv_heads*n_repeat,T,head_dim)
        V_expanded = self._repeat_kv(self.n_repeat, V)

        # Q_f32=Q.to(torch.float32)
        # K_expanded_f32=K_expanded.to(torch.float32)
        # # 4.Attention Score
        # attn_score = Q_f32 @ K_expanded_f32.transpose(-1, -2) / self.head_dim**0.5
        # attn_score=attn_score.to(Q.dtype)
        # # (B,n_head,T,T)

        # # masking: create (1,1,T_query,T_kv) mask and broadcast
        # T_kv = K_expanded.size(2)
        # if kv_cache is not None and T_kv > 1:
        #     #Inference # No masking needed
        #     #Training Standard
        #     mask = self.mask[:T, :T_kv]
        #     # ~mask is bitwise NOT (logical NOT) applied to a boolean tensor
        #     attn_score = attn_score.masked_fill(
        #         ~mask.unsqueeze(0).unsqueeze(0), float("-inf")
        #     )

        # attn_weights = F.softmax(attn_score, dim=-1)

        # attn_weights = self.droupout(attn_weights)
        # # (B,n_head,T,T) @ (B,n_head,T,head_dim) -> (B,n_head,T,head_dim)
        # out = attn_weights @ V_expanded
        is_causal = True if kv_cache is None else False
        out = F.scaled_dot_product_attention(
            query=Q,
            key=K_expanded,
            value=V_expanded,
            attn_mask=None,
            dropout_p=self.droupout.p if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        return out


class TranformerBLock(nn.Module):
    def __init__(
        self,
        context_length,
        dropout,
        n_kv_heads,
        n_head,
        d_model,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.GQA = GroupQueryAttention(
            d_model, n_head, context_length, dropout, n_kv_heads
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x, pos: torch.Tensor, kv_cache: KVCache = None):

        x = x + self.GQA(self.ln1(x), pos=pos, kv_cache=kv_cache)
        x = x + self.ff(self.ln2(x))
        return x


class TinyLM(nn.Module):
    def __init__(
        self,
        context_length,
        dropout,
        n_layer,
        n_kv_heads,
        n_head,
        d_model,
        vocab_size=None,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.embd = nn.Embedding(vocab_size, d_model)
        self.layer = nn.ModuleList(
            [
                TranformerBLock(context_length, dropout, n_kv_heads, n_head, d_model)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        # Add Weight tying
        self.lm_head.weight = self.embd.weight
        self.apply((self._init_weight))

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, y=None, kv_caches: list = None):
        """
        x: (B, T)
        kv_caches: list of KVCache objects of length n_layer (or None).
                   If provided, they will be passed into each layer's attention.
        """
        B, T = x.shape
        if kv_caches is not None and T == 1 and kv_caches[0].key is not None:
            # Inference Mode current position is (T_history + current token index)
            # Shift the position index by the length of the existing cache
            T_history = kv_caches[0].key.size(2)
            # pos for RoPE: (1)
            pos = torch.tensor([T_history], dtype=torch.long, device=device)
        else:
            # Training mode or first step of inference, pos is (0, 1, ..., T-1)
            # pos for RoPE: (T,)
            pos = torch.arange(0, T, dtype=torch.long, device=device)

        # 3.Token Embedding
        x = self.embd(x)
        for i, layer in enumerate(self.layer):
            cache = kv_caches[i] if kv_caches is not None else None
            x = layer(x, pos=pos, kv_cache=cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if y is not None:
            logits = logits.view(-1, vocab_size)
            loss = F.cross_entropy(logits, y.view(-1))
            return logits, loss
        return logits, loss

    @torch.no_grad
    def generate(self, idx, max_new_tokens=100, temperature=0.8):
        self.eval()
        idx = (
            torch.tensor(encode_text(idx), dtype=torch.long)
            .unsqueeze(0)
            .to(device=device)
        )  # (1,T)

        kv_caches = [KVCache() for _ in range(self.n_layer)]
        logits, _ = self(idx, kv_caches=kv_caches)
        for _ in range(max_new_tokens):
            #scaling by temperature
            logits_last = logits[:, -1, :] / temperature
            probs = F.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).to(device)
            logits, _ = self(next_token, kv_caches=kv_caches)
            idx = torch.cat((idx, next_token), dim=-1)
        out = decode_text(idx.squeeze(0).tolist())
        return out


learning_rate = 3e-4  # Max LR
min_lr = 6e-5  # Min LR
warmup_iters = 100  # How many steps to warm up
lr_decay_iters = 8000  # Total steps (should match your epochs)


import math


def get_lr(it):
    # 1.learning rate warm up
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2 it less than lr_decay_iters
    if it > lr_decay_iters:
        return min_lr
    # 3 Cosine decay rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1, "Decay ratio must be between 0 & 1"
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))  # 0-1 value
    return min_lr + coeff * (learning_rate - min_lr)


model = TinyLM(
    vocab_size=vocab_size,
    dropout=dropout,
    n_layer=n_layer,
    n_kv_heads=n_kv_heads,
    n_head=n_head,
    d_model=d_model,
    context_length=context_length,
).to(device=device)
# model = torch.compile(model)
# optimizer = torch.optim.AdamW(params=model.parameters())

import inspect


def configure_optimizers(model: TinyLM, weight_decay, learning_rate, device_type):
    param_dic = {pn: p for pn, p in model.named_parameters()}
    param_dic = {pn: p for pn, p in param_dic.items() if p.requires_grad}
    # Any parameter that is 2D will be weight decayed, otherwise no.
    # (i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't)
    decay_params = [p for n, p in param_dic.items() if p.dim() >= 2]
    no_decay_params = [p for n, p in param_dic.items() if p.dim() < 2]

    optim_group = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0},
    ]

    # Use fused AdamW if available (much faster on NVIDIA GPUs)
    use_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if use_fused and device_type == "cuda" else dict()
    optimizer = torch.optim.AdamW(
        optim_group, lr=learning_rate, betas=(0.9, 0.95), **extra_args
    )
    return optimizer


optimizer = configure_optimizers(
    model, weight_decay=1e-1, learning_rate=6e-4, device_type=device
)

print("Parameters:", sum(p.numel() for p in model.parameters()))

scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

# --- Configuration for Gradient Accumulation ---
micro_batch_size = B
target_Batch_size = 128
grad_accum_steps = target_Batch_size // micro_batch_size

print(f"Micro Batch: {micro_batch_size}, Accumulating {grad_accum_steps} steps.")
print(f"Effective Batch Size: {micro_batch_size * grad_accum_steps}")

optimizer.zero_grad(set_to_none=True)

import os

checkout_output = r"/kaggle/working/"
os.makedirs(checkout_output, exist_ok=True)


def save_checkpoint(step, model: TinyLM, optimizer, loss, filename="checkpoint.pth"):
    file_path = os.path.join(checkout_output, filename)
    print(f"Saving checkpoint to {file_path}")
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        file_path,
    )


def load_checkpoint(model: TinyLM, optimizer, filename="checkpoint.pth"):
    # file_path=os.path.join(checkout_output,filename)
    if os.path.exists(filename):
        print(f"Loading checkpoint from {filename}")
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["step"], checkpoint["loss"]
    return 0, None


start_step, start_loss = load_checkpoint(
    model=model,
    optimizer=optimizer,
    filename="/kaggle/input/latest-checkpoint3/best_checkpoint (2).pth",
)
best_val_loss = float("inf")
print(f"start step:{start_step},start loss:{start_loss}")
epochs = 8000

for step in range(start_step, epochs):
    # 1. Gradient Accumulation Loop
    for micro_step in range(grad_accum_steps):

        x, y = dataloader.get_batch("train")
        x, y = x.to(device), y.to(device)

        # 1. AMP Context Manager (Automatic Mixed Precision)
        # When to use lower precision (float16):Matrix Multiplication
        # When to use Higher precision (float32):softmax layer,gradient calculation
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits, loss = model(x, y)
            loss = loss / grad_accum_steps
        # Accumulate gradients (does NOT update weights yet)
        # 2. Scale the loss before backward pass
        scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  # Unscale gradients to clip them correctly
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # update the weights
    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad(set_to_none=True)

    if step % 100 == 0:
        losses = estimate_loss(micro_batch_size)
        print(f"Step {step}, train :{losses['train']:.4f}  Val: {losses['val']:.4f}")

        with open("/kaggle/working/losses.csv", "a", encoding="utf-8") as f:
            f.write(
                f"Step {step}, train :{losses['train']:.4f}  Val: {losses['val']:.4f}\n"
            )
            f.flush()

        # Save the "latest" version to resume if it crashes
        save_checkpoint(
            step=step,
            model=model,
            optimizer=optimizer,
            loss=loss,
            filename="latest_checkpoint.pth",
        )

        # Calculate validation loss to see if this is the "best" model
        if losses["val"] < best_val_loss:
            save_checkpoint(
                step=step,
                model=model,
                optimizer=optimizer,
                loss=loss,
                filename="best_checkpoint.pth",
            )
            best_val_loss = losses["val"]
            print(f"New best model saved! Loss: {losses['val']:.4f}")


prompt = "नेपाल "
newStory = model.generate(prompt)
print("\n--- Generated Text ---")
print(newStory)
