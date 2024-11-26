import torch
import pytest
import numpy as np
from src.activation_pruning import ActivationBasedPruning

class TestActivationPruning:
    def test_initialization(self, setup_pruning):
        """Test proper initialization of pruning class"""
        pruner = setup_pruning
        assert hasattr(pruner, 'model')
        assert hasattr(pruner, 'tokenizer')
        assert hasattr(pruner, 'dataset')
        assert pruner.importance_scores is None

    def test_head_dimensions_preserved(self, setup_pruning):
        """Test that head pruning preserves expected tensor dimensions"""
        pruner = setup_pruning
        original_dims = {
            'qkv': pruner.model.transformer.h[0].attn.qkv_proj.weight.shape,
            'out': pruner.model.transformer.h[0].attn.out_proj.weight.shape
        }
        
        pruner.collect_activations(batch_size=1)
        pruner.prune_model({'heads': 1})  # Prune 1 head
        
        assert pruner.model.transformer.h[0].attn.qkv_proj.weight.shape == original_dims['qkv']
        assert pruner.model.transformer.h[0].attn.out_proj.weight.shape == original_dims['out']

    def test_neuron_dimensions_preserved(self, setup_pruning):
        """Test that MLP pruning preserves expected tensor dimensions"""
        pruner = setup_pruning
        original_dims = {
            'fc_in': pruner.model.transformer.h[0].mlp.fc_in.weight.shape,
            'fc_out': pruner.model.transformer.h[0].mlp.fc_out.weight.shape
        }
        
        pruner.collect_activations(batch_size=1)
        pruner.prune_model({'neurons': 50})  # Prune 50 neurons
        
        assert pruner.model.transformer.h[0].mlp.fc_in.weight.shape == original_dims['fc_in']
        assert pruner.model.transformer.h[0].mlp.fc_out.weight.shape == original_dims['fc_out']

    def test_head_actually_pruned(self, setup_pruning):
        """Test that pruned heads are actually zeroed out"""
        pruner = setup_pruning
        pruner.collect_activations(batch_size=1)
        pruned_info = pruner.prune_model({'heads': 1})
        
        pruned_layer, pruned_head = pruned_info['heads'][0]
        layer = pruner.model.transformer.h[pruned_layer].attn
        head_size = pruner.model.config.n_embd // pruner.model.config.n_head
        start_idx = pruned_head * head_size
        end_idx = start_idx + head_size
        
        # Add debugging for the test
        print("\nTest verification:")
        print(f"Checking slice {start_idx}:{end_idx}")
        print(f"Sample values in test: {layer.qkv_proj.weight[:5, start_idx:end_idx]}")
        print(f"Max value in test: {layer.qkv_proj.weight[:, start_idx:end_idx].abs().max()}")
        
        # Check Q, K, V projections
        qkv_weights = layer.qkv_proj.weight
        for offset in [0, head_size, 2 * head_size]:  # For Q, K, and V
            slice_values = qkv_weights[:, start_idx + offset:end_idx + offset]
            print(f"\nChecking QKV slice with offset {offset}:")
            print(f"Max value: {slice_values.abs().max()}")
            assert torch.all(slice_values == 0)

    def test_neurons_actually_pruned(self, setup_pruning):
        """Test that pruned neurons are actually zeroed out"""
        pruner = setup_pruning
        pruner.collect_activations(batch_size=1)
        pruned_info = pruner.prune_model({'neurons': 1})
        
        pruned_layer, pruned_neuron = pruned_info['neurons'][0]
        layer = pruner.model.transformer.h[pruned_layer].mlp
        
        assert torch.all(layer.fc_in.weight[pruned_neuron, :] == 0)
        assert torch.all(layer.fc_out.weight[:, pruned_neuron] == 0)

    def test_activation_collection(self, setup_pruning):
        """Test activation collection process"""
        pruner = setup_pruning
        pruner.collect_activations(batch_size=1)
        
        # Check that activations were collected for all layers
        assert len(pruner.importance_scores['heads']) == len(pruner.model.transformer.h)
        assert len(pruner.importance_scores['neurons']) == len(pruner.model.transformer.h)
        
        # Check shapes
        for layer_idx, scores in pruner.importance_scores['heads'].items():
            assert len(scores) == pruner.model.config.n_head
            
        for layer_idx, scores in pruner.importance_scores['neurons'].items():
            mlp_size = pruner.model.transformer.h[0].mlp.fc_in.weight.shape[0]
            assert len(scores) == mlp_size

    def test_invalid_pruning_amounts(self, setup_pruning):
        """Test that invalid pruning amounts raise appropriate errors"""
        pruner = setup_pruning
        pruner.collect_activations(batch_size=1)
        
        # Test negative number of heads
        with pytest.raises(ValueError):
            pruner.prune_model({'heads': -1})
        
        # Test too many heads
        with pytest.raises(ValueError):
            pruner.prune_model({'heads': 1000})
        
        # Test too many neurons
        with pytest.raises(ValueError):
            total_neurons = pruner.model.transformer.h[0].mlp.fc_in.weight.shape[0] * len(pruner.model.transformer.h)
            pruner.prune_model({'neurons': total_neurons + 1})

    def test_pruning_without_activation_collection(self, setup_pruning):
        """Test that pruning automatically collects activations if needed"""
        pruner = setup_pruning
        assert pruner.importance_scores is None
        pruner.prune_model({'heads': 1})
        assert pruner.importance_scores is not None  # Should have auto-collected

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_different_batch_sizes(self, setup_pruning, batch_size):
        """Test that activation collection works with different batch sizes"""
        pruner = setup_pruning
        pruner.collect_activations(batch_size=batch_size)
        assert len(pruner.importance_scores['heads']) > 0
        assert len(pruner.importance_scores['neurons']) > 0 