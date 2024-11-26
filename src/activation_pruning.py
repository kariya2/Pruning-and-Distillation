import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from typing import Dict, List, Callable, Optional, Any, Tuple

class ActivationBasedPruning:
    def __init__(
        self, 
        model: nn.Module,
        tokenizer,
        dataset,
        aggregation_fn: Optional[Callable] = None,
        calibration_dataset_size: int = 1024
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.importance_scores = None
        self.calibration_dataset_size = calibration_dataset_size
        
        # Default to L2 norm if no aggregation function specified
        self.aggregation_fn = aggregation_fn or self._l2_norm
    
    @staticmethod
    def _l2_norm(tensor: torch.Tensor, dims: tuple) -> torch.Tensor:
        """Compute L2 norm across specified dimensions."""
        return torch.norm(tensor, p=2, dim=dims)
    
    @staticmethod
    def _mean_abs(tensor: torch.Tensor, dims: tuple) -> torch.Tensor:
        """Compute mean absolute value across specified dimensions."""
        return torch.mean(tensor.abs(), dim=dims)
    
    @staticmethod
    def _variance(tensor: torch.Tensor, dims: tuple) -> torch.Tensor:
        """Compute variance across specified dimensions."""
        return torch.var(tensor, dim=dims)
    
    def prepare_prompts(self, samples: List[Dict[str, Any]]) -> List[str]:
        """Prepare MBPP-style prompts for a batch"""
        return [
            f"""# Write a Python function for this task. Only the function, no tests or examples:
# {sample['text']}

def solution""" 
            for sample in samples
        ]
    
    def collect_activations(self, batch_size: int = 8):
        """Collect activation statistics from forward passes."""
        samples = self.dataset.select(range(min(self.calibration_dataset_size, len(self.dataset))))
        activation_stats = defaultdict(list)
        
        def hook_fn(layer_name: str):
            def hook(module, input, output):
                if layer_name.startswith('mlp'):
                    # Get activations after fc_in and activation
                    intermediate_output = module.act(module.fc_in(input[0]))  # [batch, seq, 4096]
                    # Aggregate across sequence length
                    score = self.aggregation_fn(intermediate_output, dims=1)  # [batch, 4096]
                    activation_stats[layer_name].append(score.detach().cpu())
                else:  # attention head case remains the same
                    if isinstance(output, tuple):
                        attn_output = output[0]
                    else:
                        attn_output = output
                    B, S, D = attn_output.shape
                    H = module.num_attention_heads
                    head_dim = D // H
                    attn_output = attn_output.view(B, S, H, head_dim)
                    score = self.aggregation_fn(attn_output, dims=(1, -1))
                    activation_stats[layer_name].append(score.detach().cpu())
            return hook
        
        # Register hooks
        hooks = []
        for idx, layer in enumerate(self.model.transformer.h):
            hooks.append(layer.attn.register_forward_hook(hook_fn(f'attn_{idx}')))
            hooks.append(layer.mlp.register_forward_hook(hook_fn(f'mlp_{idx}')))
        
        try:
            # Batch inference
            self.model.eval()
            with torch.no_grad():
                for i in tqdm(range(0, len(samples), batch_size), desc="Collecting activations"):
                    batch = samples.select(range(i, min(i + batch_size, len(samples))))
                    # Prepare prompts consistently
                    prompts = self.prepare_prompts(batch)
                    
                    # Handle tokenizer padding consistently
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                    
                    inputs = self.tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                        padding_side='left'
                    ).to(self.device)
                    
                    _ = self.model(**inputs)
            
            # Process activation statistics
            self.importance_scores = {
                'heads': {},
                'neurons': {}
            }
            
            # Average across batches
            for layer_idx in range(len(self.model.transformer.h)):
                # Process attention heads
                attn_stats = torch.cat(activation_stats[f"attn_{layer_idx}"], dim=0)
                self.importance_scores['heads'][layer_idx] = attn_stats.mean(0).tolist()
                
                # Process MLP neurons
                mlp_stats = torch.cat(activation_stats[f"mlp_{layer_idx}"], dim=0)
                self.importance_scores['neurons'][layer_idx] = mlp_stats.mean(0).tolist()
                
        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()
    
    def prune_model(self, prune_specs: Dict[str, int]) -> Dict[str, List[Tuple[int, int]]]:
        """
        Prune model based on activation statistics.
        Args:
            prune_specs: Dict specifying number of components to prune
                {
                    'heads': num_heads_to_prune,
                    'neurons': num_neurons_to_prune
                }
        Returns:
            Dict containing lists of (layer_idx, component_idx) for pruned heads and neurons
        """
        if not self.importance_scores:
            self.collect_activations()
        
        # Validate pruning amounts
        if 'heads' in prune_specs:
            if prune_specs['heads'] < 0:
                raise ValueError("Cannot prune negative number of heads")
            total_heads = self.model.config.n_head * len(self.model.transformer.h)
            if prune_specs['heads'] > total_heads:
                raise ValueError(f"Cannot prune {prune_specs['heads']} heads, only {total_heads} available")
        
        if 'neurons' in prune_specs:
            if prune_specs['neurons'] < 0:
                raise ValueError("Cannot prune negative number of neurons")
            neurons_per_layer = self.model.transformer.h[0].mlp.fc_in.weight.shape[0]
            total_neurons = neurons_per_layer * len(self.model.transformer.h)
            if prune_specs['neurons'] > total_neurons:
                raise ValueError(f"Cannot prune {prune_specs['neurons']} neurons, only {total_neurons} available")
        
        pruned_components = {'heads': [], 'neurons': []}
        
        if 'heads' in prune_specs:
            pruned_components['heads'] = self._prune_heads(prune_specs['heads'])
        
        if 'neurons' in prune_specs:
            pruned_components['neurons'] = self._prune_neurons(prune_specs['neurons'])
        
        return pruned_components
    
    def _prune_heads(self, num_heads: int) -> List[Tuple[int, int]]:
        """Prune specified number of attention heads."""
        head_scores = [
            (layer_idx, head_idx, score)
            for layer_idx, scores in self.importance_scores['heads'].items()
            for head_idx, score in enumerate(scores)
        ]
        
        head_scores.sort(key=lambda x: x[2])
        pruned_heads = []
        
        for layer_idx, head_idx, _ in head_scores[:num_heads]:
            layer = self.model.transformer.h[layer_idx]
            head_size = layer.attn.head_dim
            start_idx = head_idx * head_size
            end_idx = start_idx + head_size
            
            # For QKV projection
            qkv_weight = layer.attn.qkv_proj.weight
            mask = torch.ones_like(qkv_weight, dtype=bool)
            
            # Zero out Q,K,V sections for this head
            for offset in [0, head_size, 2 * head_size]:
                start = start_idx + offset
                end = end_idx + offset
                mask[:, start:end] = 0
                
            prune.CustomFromMask.apply(
                layer.attn.qkv_proj, 
                'weight',
                mask
            )
            
            # For output projection
            out_weight = layer.attn.out_proj.weight
            mask = torch.ones_like(out_weight, dtype=bool)
            mask[start_idx:end_idx, :] = 0
            
            prune.CustomFromMask.apply(
                layer.attn.out_proj, 
                'weight',
                mask
            )
            
            # Make pruning permanent
            
            pruned_heads.append((layer_idx, head_idx))
        
        return pruned_heads
    
    def _prune_neurons(self, num_neurons: int) -> List[Tuple[int, int]]:
        """Prune specified number of MLP neurons using PyTorch pruning primitives."""
        neuron_scores = [
            (layer_idx, neuron_idx, score)
            for layer_idx, scores in self.importance_scores['neurons'].items()
            for neuron_idx, score in enumerate(scores)
        ]
        
        neuron_scores.sort(key=lambda x: x[2])
        pruned_neurons = []
        
        for layer_idx, neuron_idx, _ in neuron_scores[:num_neurons]:
            layer = self.model.transformer.h[layer_idx].mlp
            
            # Create mask for input weights (1s for weights to keep, 0s for weights to prune)
            in_mask = torch.ones_like(layer.fc_in.weight, dtype=bool)
            in_mask[neuron_idx, :] = 0  # Prune the neuron's input weights
            
            # Create mask for output weights
            out_mask = torch.ones_like(layer.fc_out.weight, dtype=bool)
            out_mask[:, neuron_idx] = 0  # Prune the neuron's output weights
            
            # Apply masks using CustomFromMask
            prune.CustomFromMask.apply(layer.fc_in, 'weight', in_mask)
            prune.CustomFromMask.apply(layer.fc_out, 'weight', out_mask)
            
            pruned_neurons.append((layer_idx, neuron_idx))
        
        return pruned_neurons
