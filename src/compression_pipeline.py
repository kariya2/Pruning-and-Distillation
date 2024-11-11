import os
import json
from typing import Dict, Any, List, Union
import torch
from datetime import datetime
from pathlib import Path

from activation_pruning import ActivationBasedPruning
from evaluators.mbpp_evaluator import MBPPEvaluator
from data_loader import load_dataset

class CompressionPipeline:
    """Pipeline for iterative model compression experiments."""
    
    def __init__(
        self,
        model,
        tokenizer,
        experiment_name: str,
        base_output_dir: str = "experiments",
        eval_dataset: str = "mbpp",
        eval_split: str = "test",
        eval_batch_size: int = 16,
        eval_sample_size: int = None
    ):
        """
        Initialize compression pipeline.
        
        Args:
            model: The model to compress
            tokenizer: Model tokenizer
            experiment_name: Name for this experiment run
            base_output_dir: Base directory for saving experiment results
            eval_dataset: Dataset to use for evaluation
            eval_split: Dataset split to use
            eval_batch_size: Batch size for evaluation
            eval_sample_size: Number of samples to use for evaluation (None for all)
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Setup experiment directories
        self.experiment_dir = self._setup_experiment_dir(base_output_dir, experiment_name)
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.results_dir = self.experiment_dir / "results"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize evaluation components
        self.eval_dataset = load_dataset(eval_dataset, split=eval_split, sample_size=eval_sample_size)
        self.evaluator = MBPPEvaluator(model, tokenizer, batch_size=eval_batch_size)
        
        # Track experiment history
        self.history = []
        self.current_step = 0
        
    def _setup_experiment_dir(self, base_dir: str, experiment_name: str) -> Path:
        """Create and return experiment directory with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return experiment_dir
    
    def evaluate_model(self, step_name: str) -> Dict[str, Any]:
        """
        Evaluate current model state.
        
        Args:
            step_name: Name of the current pipeline step
        Returns:
            Dictionary containing evaluation results
        """
        print(f"\nEvaluating model after {step_name}...")
        results = self.evaluator.run_evaluation(self.eval_dataset, k=[1, 5, 10])
        
        # Save results
        results_file = self.results_dir / f"eval_results_step_{self.current_step}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'step': self.current_step,
                'step_name': step_name,
                'results': results
            }, f, indent=2)
            
        return results
    
    def save_checkpoint(self, step_name: str):
        """
        Save model checkpoint and experiment state.
        
        Args:
            step_name: Name of the current pipeline step
        """
        checkpoint_dir = self.checkpoints_dir / f"step_{self.current_step}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir / "model")
        self.tokenizer.save_pretrained(checkpoint_dir / "model")
        
        # Save experiment state
        state = {
            'step': self.current_step,
            'step_name': step_name,
            'history': self.history
        }
        with open(checkpoint_dir / "state.json", 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_checkpoint(self, step: int):
        """
        Load model and experiment state from checkpoint.
        
        Args:
            step: Step number to load
        """
        checkpoint_dir = self.checkpoints_dir / f"step_{step}"
        if not checkpoint_dir.exists():
            raise ValueError(f"No checkpoint found for step {step}")
        
        # Load model
        self.model = self.model.from_pretrained(
            checkpoint_dir / "model",
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = self.tokenizer.from_pretrained(checkpoint_dir / "model")
        
        # Load experiment state
        with open(checkpoint_dir / "state.json", 'r') as f:
            state = json.load(f)
        self.current_step = state['step']
        self.history = state['history']
    
    def run_pruning_step(
        self,
        prune_specs: Dict[str, Union[int, float]],
        calibration_size: int = 1024
    ):
        """
        Run activation-based pruning step.
        
        Args:
            prune_specs: Specifications for pruning each component
            calibration_size: Number of samples to use for calibration
        """
        step_name = f"pruning_{self.current_step}"
        print(f"\nRunning pruning step {self.current_step}...")
        
        # Initialize pruning
        pruner = ActivationBasedPruning(
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=self.eval_dataset,
            calibration_dataset_size=calibration_size
        )
        
        # Run pruning
        pruner.importance_scores = pruner.compute_importance_scores()
        pruner.prune_model(prune_specs)
        
        # Evaluate and save results
        eval_results = self.evaluate_model(step_name)
        self.save_checkpoint(step_name)
        
        # Update history
        self.history.append({
            'step': self.current_step,
            'type': 'pruning',
            'specs': prune_specs,
            'results': eval_results
        })
        self.current_step += 1
        
        return eval_results
    
    def run_iterative_pruning(
        self,
        prune_schedule: List[Dict[str, Union[int, float]]],
        calibration_size: int = 1024
    ):
        """
        Run multiple rounds of pruning according to schedule.
        
        Args:
            prune_schedule: List of pruning specifications for each round
            calibration_size: Number of samples to use for calibration
        """
        print(f"Starting iterative pruning with {len(prune_schedule)} rounds...")
        
        results = []
        for round_idx, prune_specs in enumerate(prune_schedule):
            print(f"\nPruning round {round_idx + 1}/{len(prune_schedule)}")
            round_results = self.run_pruning_step(
                prune_specs=prune_specs,
                calibration_size=calibration_size
            )
            results.append(round_results)
            
        return results
    
    def load_all_results(self):
        """Load all evaluation results from the experiment."""
        results = []
        for result_file in sorted(self.results_dir.glob("eval_results_step_*.json")):
            with open(result_file, 'r') as f:
                results.append(json.load(f))
        return results
    
    def get_experiment_summary(self):
        """Generate summary of experiment results."""
        results = self.load_all_results()
        summary = {
            'experiment_name': self.experiment_dir.name,
            'steps': []
        }
        
        for result in results:
            step_summary = {
                'step': result['step'],
                'step_name': result['step_name'],
                'pass@k': result['results']['pass@k'],
                'mean_time': result['results'].get('mean_time', None)
            }
            summary['steps'].append(step_summary)
        
        return summary

def main():
    """Example usage of compression pipeline."""
    from data_loader import load_model
    
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load model
    model_name = "Salesforce/codegen-350M-mono"
    tokenizer, model = load_model(model_name)
    
    # Initialize pipeline
    pipeline = CompressionPipeline(
        model=model,
        tokenizer=tokenizer,
        experiment_name="codegen350M_progressive_pruning",
        eval_sample_size=500  # Increased for better evaluation
    )
    
    # Progressive pruning schedule
    pruning_schedule = [
        # Round 1: Light pruning (10%)
        {
            'heads': 32,    # 10% of heads (320 total)
            'neurons': 400,  # ~10% of neurons per layer
            'embeddings': 0  # No embedding pruning initially
        },
        # Round 2: Moderate pruning (20%)
        {
            'heads': 64,    # 20% of heads
            'neurons': 800,  # ~20% of neurons
            'embeddings': 100  # Light embedding pruning
        },
        # Round 3: Aggressive pruning (30%)
        {
            'heads': 96,     # 30% of heads
            'neurons': 1200,  # ~30% of neurons
            'embeddings': 200 # Moderate embedding pruning
        },
        # Round 4: Final pruning (40%)
        {
            'heads': 128,    # 40% of heads
            'neurons': 1600, # ~40% of neurons
            'embeddings': 300 # Aggressive embedding pruning
        }
    ]
    
    # Run pruning experiments with larger calibration set
    results = pipeline.run_iterative_pruning(
        prune_schedule=pruning_schedule,
        calibration_size=256  # Increased for better calibration
    )
    
    # Print final summary
    summary = pipeline.get_experiment_summary()
    print("\nExperiment Summary:")
    for step in summary['steps']:
        print(f"\nStep {step['step']} ({step['step_name']}):")
        print(f"Pass@1: {step['pass@k']['1']:.2f}%")
        print(f"Pass@5: {step['pass@k']['5']:.2f}%")
        print(f"Pass@10: {step['pass@k']['10']:.2f}%")
    
    print("\nExperiment complete! Results saved in:", pipeline.experiment_dir)

if __name__ == "__main__":
    main() 