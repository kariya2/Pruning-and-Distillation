import os
from datasets import load_dataset
from data_loader import load_model, load_dataset
from evaluators.mbpp_evaluator import MBPPEvaluator
import json
from datetime import datetime
from glob import glob
import re

def filter_simple_problems(dataset):
    """Filter for problems containing specific keywords"""
    return dataset.filter(
        lambda x: any(word in x['text'].lower() 
                     for word in ['reverse', 'sum', 'count', 'average', 'maximum', 'minimum'])
    )

def get_next_run_id(results_dir: str, model_name: str, dataset_name: str) -> int:
    """Get the next available run ID for the given model and dataset"""
    pattern = f"{model_name}_{dataset_name}_run_*.json"
    existing_files = glob(os.path.join(results_dir, pattern))
    
    if not existing_files:
        return 1
        
    run_ids = []
    for f in existing_files:
        match = re.search(r'run_(\d+)\.json$', f)
        if match:
            run_ids.append(int(match.group(1)))
            
    return max(run_ids) + 1 if run_ids else 1

def save_results(results: dict, model_name: str, dataset_name: str, custom_name: str = None):
    """
    Save evaluation results to file
    Args:
        results: Results dictionary
        model_name: Name of the model
        dataset_name: Name of the dataset
        custom_name: Optional custom name for the results file
    """
    os.makedirs('results', exist_ok=True)
    
    if custom_name:
        filename = f'results/{model_name}_{dataset_name}_{custom_name}.json'
    else:
        run_id = get_next_run_id('results', model_name, dataset_name)
        filename = f'results/{model_name}_{dataset_name}_run_{run_id}.json'
    
    # Add metadata to results
    results['metadata'] = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'run_id': run_id if not custom_name else None,
        'custom_name': custom_name
    }
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")

def load_results(model_name: str = None, dataset_name: str = None, 
                run_id: int = None, custom_name: str = None) -> dict:
    """
    Load evaluation results from file
    Args:
        model_name: Optional filter by model name
        dataset_name: Optional filter by dataset name
        run_id: Optional specific run ID to load
        custom_name: Optional custom name of results file to load
    Returns:
        Dictionary containing results, or list of results if no specific file is specified
    """
    if not os.path.exists('results'):
        raise FileNotFoundError("No results directory found")
    
    if custom_name:
        pattern = f"*_{custom_name}.json"
    elif all([model_name, dataset_name, run_id]):
        pattern = f"{model_name}_{dataset_name}_run_{run_id}.json"
    elif all([model_name, dataset_name]):
        pattern = f"{model_name}_{dataset_name}_*.json"
    elif model_name:
        pattern = f"{model_name}_*.json"
    else:
        pattern = "*.json"
    
    matching_files = glob(os.path.join('results', pattern))
    
    if not matching_files:
        raise FileNotFoundError(f"No results found matching pattern: {pattern}")
    
    if len(matching_files) == 1 or run_id or custom_name:
        with open(matching_files[0], 'r') as f:
            return json.load(f)
    
    # If multiple files match, return them all as a list
    results = []
    for file in matching_files:
        with open(file, 'r') as f:
            results.append(json.load(f))
    return results

def main():
    # Load model and tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load model and tokenizer
    model_name = "Salesforce/codegen-2B-mono"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Load MBPP dataset
    dataset = load_dataset("mbpp", split="test")
    simple_problems = filter_simple_problems(dataset)
    
    # Initialize evaluator
    evaluator = MBPPEvaluator(model, tokenizer)
    
    # Run evaluation
    print("Running evaluation...")
    results = evaluator.run_evaluation(simple_problems, k=[1, 5, 10])
    print(results)
    
    # Save results (you can use either method)
    # Method 1: Auto-incrementing run ID
    save_results(results, "codegen-2B-mono", "mbpp_simple")
    
    # Method 2: Custom name
    # save_results(results, "codegen-2B-mono", "mbpp_simple", custom_name="baseline_run")
    
    # Print summary
    print(f"\nEvaluation Summary:")
    print(f"Total samples: {results['total_samples']}")
    print(f"Passed samples: {results['passed_samples']}")
    print(f"Pass rate: {results['pass_rate']*100:.2f}%")

if __name__ == "__main__":
    main() 