# CHIMERA-M: Multi-GPU Architecture

**Status:** Design Phase  
**Target:** Post-v1.0 Release  
**Scope:** Distributed training across 2-256 GPUs with adaptive resource allocation

---

## Executive Summary

Multi-GPU CHIMERA-M extends the single-device gearshift system to a distributed coordinator that synchronizes compression levels across workers while minimizing communication overhead. The core insight: **not all GPUs need the same gear**.

Traditional data-parallel training wastes bandwidth synchronizing full gradients. CHIMERA-M uses **asymmetric gears** where slow GPUs run higher compression (less data to send) while fast GPUs run lower compression (more precise updates).

---

## 1. System Architecture

### 1.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CHIMERA-M Multi-GPU Cluster                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │   GPU 0      │◄───►│   GPU 1      │◄───►│   GPU 2      │  ...            │
│  │  (Gear 2)    │     │  (Gear 3)    │     │  (Gear 2)    │                 │
│  │  Ternary     │     │  BF16        │     │  Ternary     │                 │
│  │  8GB VRAM    │     │  16GB VRAM   │     │  8GB VRAM    │                 │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘                 │
│         │                    │                    │                           │
│         └────────────────────┴────────────────────┘                           │
│                         │                                                    │
│                         ▼                                                    │
│              ┌─────────────────────┐                                         │
│              │  Gear Coordinator   │                                         │
│              │  (Global Consensus) │                                         │
│              │  - Monitors all GPUs│                                         │
│              │  - Decides global gear│                                        │
│              │  - Handles stragglers │                                        │
│              └─────────────────────┘                                         │
│                         │                                                    │
│                         ▼                                                    │
│              ┌─────────────────────┐                                         │
│              │  Async Gradient     │                                         │
│              │  Buckets by Gear    │                                         │
│              │  - Gear 1: Full     │                                         │
│              │  - Gear 3: Sketch   │                                         │
│              │  - Gear 5: Sparse   │                                         │
│              └─────────────────────┘                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Components

| Component | Responsibility | Location |
|-----------|--------------|----------|
| **Local Gearshift** | Per-GPU watchdog, monitors VRAM/temperature | Each GPU worker |
| **Gear Coordinator** | Global gear consensus, handles stragglers | Rank 0 or separate process |
| **Gradient Router** | Routes updates by gear level, applies compression | Each GPU |
| **Async Aggregator** | Non-blocking all-reduce with gear-specific bucketing | Distributed |
| **State Synchronizer** | Checkpoint consistency across shards | Rank 0 |

---

## 2. Distributed Gearshift Protocol

### 2.1 Gear Consensus Algorithm

The challenge: GPUs have different capabilities (VRAM, temperature, speed). We need a **global gear** that doesn't slow down fast GPUs or crash slow ones.

```python
class GearCoordinator:
    """
    Distributed gear consensus using Byzantine fault tolerance principles.
    
    Algorithm:
    1. Each GPU reports local gear (based on its own watchdog)
    2. Coordinator computes "safe gear" = min(reported_gears) - 1
    3. Safe gear is broadcast to all GPUs
    4. GPUs have 3 options:
       a. Accept (if local gear >= safe gear)
       b. Protest (if local gear < safe gear, requests emergency mode)
       c. Adapt (activates asymmetry, see Section 3)
    
    Failure modes:
    - Straggler GPU: Slow GPU gets isolated to Gear 5 (MEZO), others continue at Gear 2-3
    - Network partition: Sub-clusters run independently with periodic sync attempts
    - Hot GPU: Thermal throttle triggers automatic downshift without consensus
    """
    
    def __init__(self, world_size: int, fault_tolerance: float = 0.1):
        self.world_size = world_size
        self.fault_tolerance = fault_tolerance  # Allow 10% stragglers
        
    def consensus_round(self, local_gears: List[int]) -> Tuple[int, Dict]:
        """
        Compute global gear from local reports.
        
        Returns:
            global_gear: Int gear level (1-5)
            assignments: Dict[rank, gear] - per-rank gear assignments
            mode: "uniform" or "asymmetric"
        """
        # Sort gears to find percentiles
        sorted_gears = sorted(local_gears)
        
        # Find cutoff point (allow fault_tolerance fraction to be slower)
        cutoff_idx = int(self.world_size * (1 - self.fault_tolerance))
        safe_gear = sorted_gears[cutoff_idx] - 1
        safe_gear = max(1, min(5, safe_gear))  # Clamp to valid range
        
        # Assign gears
        assignments = {}
        mode = "uniform"
        
        for rank, local_gear in enumerate(local_gears):
            if local_gear <= safe_gear + 1:
                # GPU can handle safe gear
                assignments[rank] = safe_gear
            else:
                # GPU needs lower compression (asymmetry)
                assignments[rank] = local_gear
                mode = "asymmetric"
        
        return safe_gear, assignments, mode
```

### 2.2 Asymmetric Gear Mode

When GPUs have heterogeneous capabilities:

```
Scenario: 4 GPUs training 7B model
- GPU 0: A100 80GB → Gear 1 (BF16, full precision)
- GPU 1: A100 40GB → Gear 2 (Ternary base + BF16 shadow)
- GPU 2: RTX 4090 24GB → Gear 3 (Ternary + sparse updates)
- GPU 3: RTX 3090 12GB → Gear 5 (MEZO, zeroth-order)

Communication Strategy:
┌──────────┬──────────┬──────────┬──────────┐
│  GPU 0   │  GPU 1   │  GPU 2   │  GPU 3   │
├──────────┼──────────┼──────────┼──────────┤
│ BF16     │ Ternary  │ Ternary  │ MEZO     │
│ Updates  │ Updates  │ Sparse   │ 2 Samples│
│ 28GB     │ 2.3GB    │ 0.7GB    │ 0.5MB    │
│ /step    │ /step    │ /step    │ /step    │
└──────────┴──────────┴──────────┴──────────┘

Routing:
- GPU 0 sends full BF16 gradients (high bandwidth)
- GPU 1-2 send ternary-packed weights (medium bandwidth)
- GPU 3 sends only loss deltas (tiny bandwidth)

Result: 90% bandwidth reduction vs naive data-parallel
```

### 2.3 Communication Patterns by Gear

| Gear | Local Storage | Communication Pattern | Bandwidth |
|------|---------------|----------------------|-----------|
| 1 | BF16 weights + FP32 optimizer | Full gradient all-reduce | 100% |
| 2 | Ternary weights + BF16 shadow | Compressed gradient + sketch delta | 25% |
| 3 | Ternary + sparse updates | Top-K gradient indices + values | 10% |
| 4 | Ternary + SSD spill | Async checkpoint sync only | 5% |
| 5 | Ternary only, MEZO mode | Loss samples only (zeroth-order) | 0.1% |

---

## 3. Gradient Routing and Aggregation

### 3.1 Gear-Specific Gradient Buckets

```python
class GradientRouter:
    """
    Routes gradients based on gear level and destination.
    
    Each GPU maintains separate communication buffers per gear:
    - Buffer 1: Gear 1 peers (full precision)
    - Buffer 2: Gear 2-3 peers (ternary compressed)
    - Buffer 3: Gear 5 peers (MEZO samples)
    """
    
    def __init__(self, world_size: int, gear_assignments: Dict[int, int]):
        self.world_size = world_size
        self.assignments = gear_assignments
        
        # Group peers by gear
        self.gear_groups = self._group_by_gear()
        
    def _group_by_gear(self) -> Dict[int, List[int]]:
        """Group GPU ranks by their assigned gear level."""
        groups = {1: [], 2: [], 3: [], 4: [], 5: []}
        for rank, gear in self.assignments.items():
            groups[gear].append(rank)
        return groups
    
    def prepare_gradient(self, gradients: torch.Tensor, my_gear: int) -> Dict[int, Any]:
        """
        Prepare gradients for each destination gear group.
        
        Returns dict: gear_level -> compressed_data
        """
        prepared = {}
        
        for dest_gear, peers in self.gear_groups.items():
            if not peers:
                continue
                
            if dest_gear == 1:
                # Full precision for Gear 1
                prepared[dest_gear] = gradients.clone()
                
            elif dest_gear in [2, 3]:
                # Ternary compression for Gear 2-3
                codec = TernaryCodec()
                packed, meta = codec.encode(gradients)
                prepared[dest_gear] = (packed, meta)
                
            elif dest_gear == 5:
                # MEZO: only send loss, not gradients
                # (handled separately in forward pass)
                prepared[dest_gear] = None
                
        return prepared
    
    def aggregate_async(self, local_data: Dict[int, Any]) -> torch.Tensor:
        """
        Async all-reduce with gear-appropriate communication.
        
        Strategy:
        1. Start async sends to all peer groups simultaneously
        2. While waiting, decompress incoming data
        3. Weight by gear precision (Gear 1 gets 4x weight vs Gear 3)
        4. Return fused gradient
        """
        # Implementation uses torch.distributed with custom backend
        pass
```

### 3.2 Ring-Reduce with Gear-Aware Chunking

Traditional ring-reduce sends equal chunks. CHIMERA-M uses **weighted chunks** based on gear precision:

```
Ring-Reduce for 4 GPUs (7B model, ~28GB total):

Standard (naive):
Each GPU sends 7GB to next, 4 hops = 28GB per GPU transferred

CHIMERA-M Gear-Aware:
┌──────┬────────┬────────────────────────────────┐
│ GPU  │ Gear   │ Chunk Size (weighted by precision) │
├──────┼────────┼────────────────────────────────┤
│ 0    │ 1      │ 14GB (2x weight, BF16)         │
│ 1    │ 2      │ 7GB (1x weight, Ternary)       │
│ 2    │ 2      │ 7GB (1x weight, Ternary)       │
│ 3    │ 5      │ 0GB (MEZO, no gradients)       │
└──────┴────────┴────────────────────────────────┘

Total transfer per GPU: 14GB (50% reduction)
Precision-weighted aggregation maintains convergence
```

---

## 4. State Synchronization and Checkpointing

### 4.1 Sharded Optimizer States

In multi-GPU, the Count-Min Sketch becomes **sharded**:

```python
class ShardedCountMinSketch:
    """
    Distributed Count-Min Sketch where each GPU owns a shard.
    
    Instead of replicating 16KB sketch on each GPU:
    - GPU 0 owns buckets 0-255
    - GPU 1 owns buckets 256-511
    - ...
    
    For a model with P parameters across W workers:
    - Each GPU tracks P/W parameters in its local sketch
    - Cross-shard queries use all-gather (rare, batch at end of step)
    """
    
    def __init__(self, world_size: int, rank: int, 
                 global_width: int = 1024, depth: int = 4):
        self.world_size = world_size
        self.rank = rank
        
        # Each GPU gets 1/W of the buckets
        self.local_width = global_width // world_size
        self.depth = depth
        
        # Local tables
        self.tables_m = torch.zeros((depth, self.local_width))
        self.tables_v = torch.zeros((depth, self.local_width))
        
    def local_update(self, indices: torch.Tensor, values_m: torch.Tensor, values_v: torch.Tensor):
        """Update only local shard."""
        # Filter indices to this shard
        shard_mask = (indices % self.world_size) == self.rank
        local_indices = indices[shard_mask] // self.world_size
        local_m = values_m[shard_mask]
        local_v = values_v[shard_mask]
        
        # Update local sketch (same as single-GPU)
        # ... standard CMS update on local_indices ...
        
    def gather_full_estimate(self, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query momentum/variance across all shards.
        
        Steps:
        1. Query local shard
        2. All-gather results from all GPUs
        3. Take minimum across shards (Count-Min property)
        """
        # Local query
        local_m, local_v = self.query_local(indices)
        
        # All-gather (only happens at end of optimizer step)
        all_m = [torch.zeros_like(local_m) for _ in range(self.world_size)]
        all_v = [torch.zeros_like(local_v) for _ in range(self.world_size)]
        
        dist.all_gather(all_m, local_m)
        dist.all_gather(all_v, local_v)
        
        # Stack and take minimum
        m_estimates = torch.stack(all_m)
        v_estimates = torch.stack(all_v)
        
        m_hat = torch.min(m_estimates, dim=0)[0]
        v_hat = torch.min(v_estimates, dim=0)[0]
        
        return m_hat, v_hat
```

### 4.2 Consistent Checkpointing

```python
class DistributedCheckpoint:
    """
    Consistent checkpointing across asymmetric gears.
    
    Challenge: Different GPUs have different compression levels.
    Solution: Save unified representation + metadata.
    """
    
    def save(self, path: str):
        """
        Save checkpoint that can be resumed on any gear configuration.
        
        Structure:
        - model_state: Always full precision (decompressed)
        - optimizer_state: Gear metadata + compressed representations
        - gear_assignments: Which GPU was which gear
        - rng_state: Reproducibility
        """
        # 1. All GPUs decompress to FP32 for consistency
        full_weights = {}
        for name, param in self.model.named_parameters():
            if name in self.shadow_weights:
                # Decompress from ternary
                full_weights[name] = self.codec.decode(
                    self.shadow_weights[name],
                    self.shadow_meta[name]
                )
            else:
                full_weights[name] = param.clone()
        
        # 2. Rank 0 saves (others verify with hash)
        if self.rank == 0:
            torch.save({
                'model': full_weights,
                'optimizer': {
                    'sketch_tables': self.sketch.tables_m.cpu(),  # Sharded, gather first
                    'step_count': self.sketch.step_count,
                    'gear_history': self.gear_history,
                },
                'gear_assignments': self.assignments,
                'rng_state': torch.get_rng_state(),
            }, path)
        
    def load(self, path: str, new_assignments: Dict[int, int]):
        """
        Load checkpoint, potentially on different gear configuration.
        
        Handles cases:
        - Saved Gear 2, resuming on Gear 3 (recompress)
        - 4 GPUs saved, 8 GPUs resuming (reshard sketch)
        - Asymmetric gears changed (adapt compression)
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        # 1. Load full precision weights
        for name, param in self.model.named_parameters():
            param.data.copy_(checkpoint['model'][name])
        
        # 2. Recompress to current gear level
        current_gear = new_assignments[self.rank]
        self.apply_gear_compression(current_gear)
        
        # 3. Reshard sketch if world size changed
        saved_world = len(checkpoint['gear_assignments'])
        if saved_world != self.world_size:
            self._reshard_sketch(checkpoint['optimizer']['sketch_tables'])
```

---

## 5. Network Topology Optimization

### 5.1 Topology-Aware Gradient Routing

```
Typical Server Topology (DGX-style):

┌─────────────────────────────────────────────────────┐
│                    NVSwitch                          │
│     ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐    │
│     │GPU0 │GPU1 │GPU2 │GPU3 │GPU4 │GPU5 │GPU6 │GPU7 │
│     └─────┴─────┴─────┴─────┴─────┴─────┴─────┘    │
│           NVLink (900 GB/s)                          │
└─────────────────────────────────────────────────────┘
          │
          ▼ PCIe/NVLink Bridge
┌─────────────────────────────────────────────────────┐
│              Other Servers (InfiniBand)              │
│         (100-400 Gbps = 12.5-50 GB/s)               │
└─────────────────────────────────────────────────────┘

Strategy:
1. Intra-node (NVLink): Full sync every step (high bandwidth)
2. Inter-node (IB): Sparse sync every N steps (async)
3. Gear 5 GPUs: Only inter-node (no intra-node sync needed)
```

### 5.2 Hierarchical Synchronization

```python
class HierarchicalSynchronizer:
    """
    Two-level sync: intra-node fast, inter-node async.
    """
    
    def __init__(self, local_rank: int, local_world_size: int,
                 global_rank: int, global_world_size: int):
        self.local_rank = local_rank  # Within node
        self.local_world_size = local_world_size  # GPUs per node
        self.global_rank = global_rank
        self.global_world_size = global_world_size
        
    def step(self, gradients: torch.Tensor, gear: int):
        """
        Hierarchical all-reduce.
        """
        # Level 1: Intra-node (fast, every step)
        local_avg = self._intra_node_reduce(gradients)
        
        # Level 2: Inter-node (async, every N steps for gears 2-3)
        if gear <= 2 or self.step_count % 4 == 0:
            global_avg = self._inter_node_async_reduce(local_avg)
        else:
            # Use stale global average (async)
            global_avg = self._get_cached_global()
        
        return global_avg
```

---

## 6. Adaptive LoRA/Adapters (Multi-GPU)

### 6.1 Per-Layer Gear Distribution

```python
class LayerWiseGearAssignment:
    """
    Different layers can have different gears on different GPUs.
    
    Example: 24-layer model on 4 GPUs
    - Important layers (0, 12, 23): Gear 1 on all GPUs (replicated)
    - Middle layers: Sharded by gear capability
    """
    
    def assign_layers_to_gpus(self, num_layers: int, 
                            gpu_capabilities: List[Dict]) -> Dict[int, List[int]]:
        """
        Returns: layer_id -> list of GPU ranks that hold it
        """
        assignments = {}
        
        # Identify important layers (first, middle, last)
        important = [0, num_layers // 2, num_layers - 1]
        
        for layer_id in range(num_layers):
            if layer_id in important:
                # Replicate on all GPUs (Gear 1, full precision)
                assignments[layer_id] = list(range(len(gpu_capabilities)))
            else:
                # Shard by capability
                # Highest VRAM GPUs get more layers at lower gear
                sorted_gpus = sorted(
                    enumerate(gpu_capabilities),
                    key=lambda x: x[1]['vram_gb'],
                    reverse=True
                )
                
                # Round-robin assignment
                gpu_idx = layer_id % len(sorted_gpus)
                assignments[layer_id] = [sorted_gpus[gpu_idx][0]]
        
        return assignments
```

### 6.2 Adapter-Specific Communication

For LoRA/adapter training, communication is **adapter-sized** not model-sized:

```
Standard multi-GPU:
Communication per step = model_size × world_size
For 7B model, 4 GPUs: 28GB of gradient traffic

Adapter multi-GPU:
Communication per step = adapter_size × world_size
For rank-16 LoRA on 7B model: ~35MB of traffic

Speedup: 800× less communication

Gear assignment for adapters:
- Gear 1: Full adapter precision (rarely needed)
- Gear 2: Ternary adapter A matrix, BF16 B matrix
- Gear 3: Both matrices ternary
- Gear 4: Sparse adapter updates (only top 10% of ranks)
- Gear 5: Adapter frozen, only base model MEZO (extreme memory pressure)
```

---

## 7. Implementation Roadmap

### Phase 1: Foundation (v1.1)
- [ ] Implement GearCoordinator with consensus protocol
- [ ] Add async gradient bucketing infrastructure
- [ ] Test with 2-4 GPUs homogeneous (same gear)

### Phase 2: Asymmetry (v1.2)
- [ ] Asymmetric gear assignments
- [ ] Gear-specific communication paths
- [ ] Topology-aware routing (NVLink vs IB)
- [ ] Test with 4-8 GPUs heterogeneous

### Phase 3: Scale (v1.3)
- [ ] Sharded optimizer states
- [ ] Hierarchical sync (intra/inter node)
- [ ] Fault tolerance (straggler handling)
- [ ] Test with 16-64 GPUs

### Phase 4: Adapters (v1.4)
- [ ] LoRA integration
- [ ] Per-layer gear assignment
- [ ] Adapter-aware communication compression
- [ ] Test with 64-256 GPUs

---

## 8. Expected Performance

### Communication Reduction vs Naive Data-Parallel

| GPUs | Homogeneous Gear | Asymmetric Gear | Adapter Mode |
|------|------------------|-----------------|--------------|
| 4    | 75%              | 50%             | 99.9%        |
| 8    | 87.5%            | 65%             | 99.9%        |
| 32   | 97%              | 85%             | 99.9%        |
| 128  | 99%              | 95%             | 99.9%        |

*(Lower % = less communication overhead vs single-GPU)*

### Convergence Impact

- **Homogeneous Gear 2**: ~1% accuracy loss vs full precision (acceptable)
- **Asymmetric (Gear 1-3 mix)**: ~2% accuracy loss (acceptable with longer training)
- **Adapter mode**: No accuracy loss vs single-GPU adapter training

### Throughput Scaling

- **Linear scaling** up to 8 GPUs (communication not bottleneck)
- **85% scaling** at 32 GPUs
- **70% scaling** at 128 GPUs (with topology-aware routing)

---

## 9. Research Questions

1. **Optimal gear assignment policy**: Should fast GPUs help slow ones (knowledge distillation) or ignore them (async)?

2. **Dynamic rebalancing**: How often to reshuffle layer assignments based on training dynamics?

3. **Mixed precision in sync**: Does averaging BF16 (Gear 1) with Ternary (Gear 3) harm convergence?

4. **Fault tolerance**: What's the recovery cost of a GPU failure during asymmetric training?

---

## 10. References

- **ZeRO**: Rajbhandari et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
- **FSDP**: FairScale team "Fully Sharded Data Parallel"
- **Megatron-LM**: Shoeybi et al. "Megatron-LM: Training Multi-Billion Parameter Language Models"
- **DeepSpeed**: Rasley et al. "DeepSpeed: System Optimizations Enable Training Deep Learning Models"

---

*This architecture is speculative and subject to change during implementation.*
