import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from src.activation_pruning import ActivationBasedPruning

def test_pruning():
    print("Loading model and data...")
    model = AutoModelForCausalLM.from_pretrained(
        "Salesforce/codegen-350M-mono",  # Using smaller model for testing
        torch_dtype='auto',
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
    dataset = load_dataset("mbpp", split="test").select(range(10))  # Small subset for testing
    
    # Print initial model stats
    num_heads = model.transformer.h[0].attn.num_attention_heads
    hidden_dim = model.transformer.h[0].mlp.fc_in.weight.shape[0]
    print(f"\nInitial model stats:")
    print(f"Number of layers: {len(model.transformer.h)}")
    print(f"Number of heads per layer: {num_heads}")
    print(f"MLP hidden dimension: {hidden_dim}")
    
    # Initialize pruning
    print("\nInitializing pruning...")
    pruner = ActivationBasedPruning(model, tokenizer, dataset)
    
    # Collect activations and check shapes
    print("\nCollecting activations...")
    pruner.collect_activations(num_samples=5, batch_size=2)
    
    # Print importance score shapes
    print("\nImportance score shapes:")
    for layer_idx, head_scores in pruner.importance_scores['heads'].items():
        print(f"Layer {layer_idx} head scores: {len(head_scores)}")
    for layer_idx, neuron_scores in pruner.importance_scores['neurons'].items():
        print(f"Layer {layer_idx} neuron scores: {len(neuron_scores)}")
    
    # Test sample before pruning
    print("\nTesting model before pruning...")
    test_input = "def add(a, b):"
    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_before = model.generate(**inputs, max_length=50, num_return_sequences=1)
    print("Output before pruning:", tokenizer.decode(output_before[0]))
    
    # Prune model
    print("\nPruning model...")
    prune_specs = {
        'heads': 2,  # Prune 2 heads
        'neurons': 100  # Prune 100 neurons
    }
    pruner.prune_model(prune_specs)
    
    # Verify pruning
    print("\nVerifying pruning...")
    total_zeros = 0
    total_params = 0
    for name, param in model.named_parameters():
        if 'attn' in name or 'mlp' in name:
            zeros = (param == 0).sum().item()
            total = param.numel()
            total_zeros += zeros
            total_params += total
            print(f"{name}: {zeros}/{total} zeros ({zeros/total*100:.2f}%)")
    
    print(f"\nTotal pruned parameters: {total_zeros}/{total_params} ({total_zeros/total_params*100:.2f}%)")
    
    # Test sample after pruning
    print("\nTesting model after pruning...")
    with torch.no_grad():
        output_after = model.generate(**inputs, max_length=50, num_return_sequences=1)
    print("Output after pruning:", tokenizer.decode(output_after[0]))

if __name__ == "__main__":
    test_pruning() 