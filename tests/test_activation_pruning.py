import torch
import pytest
from activation_pruning import ActivationBasedPruning

def count_parameters(model):
    """Count the number of parameters in a model"""
    return sum(p.numel() for p in model.parameters())

class TestActivationPruning:
    def test_initialization(self, setup_pruning):
        """Test proper initialization of pruning class"""
        pruner = setup_pruning
        assert hasattr(pruner, 'model')
        assert hasattr(pruner, 'tokenizer')
        assert hasattr(pruner, 'dataset')
        assert pruner.importance_scores is None

    def test_activation_collection(self, setup_pruning):
        """Test that activation collection produces valid scores"""
        pruner = setup_pruning
        pruner.collect_activations(batch_size=1)
        
        # Check that we have scores for each layer
        assert len(pruner.importance_scores['heads']) == len(pruner.model.transformer.h)
        assert len(pruner.importance_scores['neurons']) == len(pruner.model.transformer.h)
        
        # Check score dimensions
        for layer_idx in range(len(pruner.model.transformer.h)):
            # Head scores should match number of heads
            assert len(pruner.importance_scores['heads'][layer_idx]) == pruner.model.config.n_head
            
            # Neuron scores should match intermediate size
            mlp_size = pruner.model.transformer.h[layer_idx].mlp.fc_in.weight.shape[0]
            assert len(pruner.importance_scores['neurons'][layer_idx]) == mlp_size

    def test_parameter_removal(self, setup_pruning):
        """Test that parameters are actually removed after pruning"""
        pruner = setup_pruning
        pruner.collect_activations(batch_size=1)
        
        # Get initial parameter count
        initial_params = count_parameters(pruner.model)
        
        # Prune heads and neurons
        pruner.prune_model({'heads': 1, 'neurons': 1})
        
        # Check that parameter count has decreased
        final_params = count_parameters(pruner.model)
        assert final_params < initial_params

    def test_invalid_pruning_amounts(self, setup_pruning):
        """Test that invalid pruning amounts raise appropriate errors"""
        pruner = setup_pruning
        pruner.collect_activations(batch_size=1)
        
        with pytest.raises(ValueError):
            pruner.prune_model({'heads': -1})
        
        with pytest.raises(ValueError):
            total_heads = pruner.model.config.n_head * len(pruner.model.transformer.h)
            pruner.prune_model({'heads': total_heads + 1})
        
        with pytest.raises(ValueError):
            total_neurons = pruner.model.transformer.h[0].mlp.fc_in.weight.shape[0] * len(pruner.model.transformer.h)
            pruner.prune_model({'neurons': total_neurons + 1})

    def test_model_still_functions(self, setup_pruning):
        """Test that model can still do forward pass after pruning"""
        pruner = setup_pruning
        pruner.collect_activations(batch_size=1)
        pruner.prune_model({'heads': 1, 'neurons': 1})
        
        # Try a forward pass
        test_input = pruner.tokenizer(
            ["def test(): pass"],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(pruner.model.device)
        
        with torch.no_grad():
            try:
                outputs = pruner.model(**test_input)
                assert outputs.logits.shape[2] == pruner.model.config.vocab_size
            except Exception as e:
                pytest.fail(f"Forward pass failed after pruning: {str(e)}")