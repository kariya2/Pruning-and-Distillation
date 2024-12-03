import os
import json
from typing import Dict, Any, List, Union
import torch
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from activation_pruning import ActivationBasedPruning
from evaluators.mbpp_evaluator import MBPPEvaluator
from data_loader import load_dataset, load_model
from distillation import run_distillation

@dataclass
class CompressionStep:
    """Configuration for a single compression step"""
    step_type: str  # 'prune' or 'distill'
    config: Dict[str, Any]
    
@dataclass
class CompressionResult:
    """Results from a compression step"""
    step_type: str
    metrics: Dict[str, Any]
    model_path: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert CompressionResult to a dictionary for serialization"""
        return {
            'step_type': self.step_type,
            'metrics': self.metrics,
            'model_path': self.model_path
        }
    
class CompressionPipeline:
    """Pipeline for iterative model compression experiments."""
    
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        evaluator,
        experiment_name: str,
        model_name: str,
        base_output_dir: str = "experiments",
    ):
        """Initialize compression pipeline.
        
        Args:
            model: The model to compress
            tokenizer: Model tokenizer
            train_dataset: Dataset for training/pruning
            eval_dataset: Dataset for evaluation
            evaluator: Evaluator instance
            experiment_name: Name for this experiment run
            model_name: Name of the original model
            base_output_dir: Base directory for saving experiment results
        """
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.evaluator = evaluator
        
        # Setup experiment directories
        self.experiment_dir = self._setup_experiment_dir(base_output_dir, experiment_name)
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.results_dir = self.experiment_dir / "results"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Track experiment history
        self.results = []
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
            'results': [r.to_dict() for r in self.results]  # Convert results to dict
        }
        with open(checkpoint_dir / "state.json", 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load pipeline state from checkpoint"""
        with open(checkpoint_path, 'r') as f:
            state = json.load(f)
            
        self.current_step = state['current_step']
        # Convert dictionaries back to CompressionResult objects
        self.results = [
            CompressionResult(
                step_type=r['step_type'],
                metrics=r['metrics'],
                model_path=r['model_path']
            )
            for r in state['results']
        ]
        self.experiment_dir = Path(state['experiment_dir'])
    
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
    
    def validate_loaded_model(self, model, model_path: str, model_type: str = "teacher"):
        """Validate a loaded model's weights and outputs."""
        print(f"\nValidating loaded {model_type} model from: {model_path}")
        
        # Check lm_head weights
        lm_head = model.lm_head
        weight = lm_head.weight.data
        bias = lm_head.bias.data if lm_head.bias is not None else None
        
        print("\nLanguage model head statistics:")
        print(f"Weight shape: {weight.shape}")
        print(f"Weight range: [{weight.min():.4f}, {weight.max():.4f}]")
        print(f"Weight mean: {weight.mean():.4f}")
        print(f"NaN in weights: {torch.isnan(weight).any().item()}")
        print(f"Inf in weights: {torch.isinf(weight).any().item()}")
        
        if bias is not None:
            print(f"\nBias range: [{bias.min():.4f}, {bias.max():.4f}]")
            print(f"Bias mean: {bias.mean():.4f}")
            print(f"NaN in bias: {torch.isnan(bias).any().item()}")
            print(f"Inf in bias: {torch.isinf(bias).any().item()}")
        
        # Test forward pass
        sample_input = self.tokenizer(
            "def test():",
            return_tensors="pt",
            truncation=True,
            max_length=32
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(**sample_input)
            logits = outputs.logits
            
        print("\nForward pass test:")
        print(f"Output shape: {logits.shape}")
        print(f"Output range: [{logits.min():.4f}, {logits.max():.4f}]")
        print(f"Output mean: {logits.mean():.4f}")
        print(f"NaN in output: {torch.isnan(logits).any().item()}")
        print(f"Inf in output: {torch.isinf(logits).any().item()}")
        
        # Raise error if any problems found
        if (torch.isnan(weight).any() or torch.isinf(weight).any() or 
            (bias is not None and (torch.isnan(bias).any() or torch.isinf(bias).any()))):
            raise ValueError(f"Loaded {model_type} model has corrupted weights")
    
    def run_step(self, step: CompressionStep, step_idx: int) -> CompressionResult:
        """Run a single compression step"""
        print(f"\n{'='*80}")
        print(f"Starting step {step_idx}: {step.step_type}")
        
        if step.step_type == "prune":
            # Pruning step - use current model
            print(f"\nPruning Specifications:")
            print(f"- Specs: {step.config['prune_spec']}")
            pruner = ActivationBasedPruning(
                model=self.model,
                tokenizer=self.tokenizer,
                dataset=self.train_dataset,
                **step.config.get("pruning_args", {})
            )
            metrics = pruner.prune_model(step.config["prune_spec"])
            print("\nPruning complete")
            
            #self.validate_model_state("pruning")
            
        elif step.step_type == "distill":
            # For distillation, we need to carefully manage teacher selection
            if step_idx == 1:  # First distillation
                # Load original model as teacher
                print("\nLoading original model as teacher for first distillation")
                teacher_model = type(self.model).from_pretrained(
                    self.model_name,
                    torch_dtype="auto",
                    device_map="auto"
                )
            else:  # Subsequent distillations
                # Use the last successful distilled model as teacher
                last_distill = None
                for prev_result in reversed(self.results):
                    if prev_result.step_type == "distill":
                        last_distill = prev_result
                        break
                
                if last_distill is None:
                    raise ValueError("No previous distilled model found for teacher")
                
                print(f"\nLoading previous distilled model as teacher from: {last_distill.model_path}")
                teacher_model = type(self.model).from_pretrained(
                    last_distill.model_path,
                    torch_dtype="auto",
                    device_map="auto"
                )
            
            # Validate teacher model
            #self.validate_loaded_model(teacher_model, 
                #"original_model" if step_idx == 1 else last_distill.model_path, 
                #"teacher")
            
            # Run distillation
            print("\nDistillation Configuration:")
            print(f"- Teacher: {teacher_model.__class__.__name__}")
            print(f"- Student: {self.model.__class__.__name__}")
            print(f"- Config: {step.config}")
            
            #self.validate_model_state("right before distillation")
            # reload model from checkpoint
            from transformers import AutoModelForCausalLM
            last_checkpoint = self.results[-1].model_path
            self.model = AutoModelForCausalLM.from_pretrained(last_checkpoint)

            metrics = run_distillation(
                teacher_model=teacher_model,
                student_model=self.model,
                tokenizer=self.tokenizer,
                dataset=self.train_dataset,
                **step.config
            )
            print("\nDistillation complete")
        
        # Save model and results
        step_dir = Path(self.experiment_dir) / f"step_{step_idx}"
        step_dir.mkdir(exist_ok=True)
        
        model_path = str(step_dir / "model")
        self.model.save_pretrained(model_path)
        print(f"\nSaved model to: {model_path}")
        
        # Evaluate model after step
        print("\nEvaluating model...")
        eval_results = self.evaluator.run_evaluation(self.eval_dataset, k=[1, 5, 10])
        
        # Save checkpoint and results
        self.save_checkpoint(f"{step.step_type}_{step_idx}")
        
        result = CompressionResult(
            step_type=step.step_type,
            metrics={
                'step_metrics': metrics,
                'eval_metrics': eval_results
            },
            model_path=model_path
        )
        
        # Save step results
        with open(step_dir / "results.json", "w") as f:
            result_dict = {
                "step": step_idx,
                "step_type": step.step_type,
                "config": step.config,
                "metrics": metrics,
                "eval_results": eval_results,
                "model_path": model_path,
                "timestamp": datetime.now().isoformat()
            }
            json.dump(result_dict, f, indent=2)
        print(f"Saved results to: {step_dir / 'results.json'}")
        
        self.results.append(result)
        self.current_step += 1
        return result
    
    def run_pipeline(self, steps: List[CompressionStep]) -> List[CompressionResult]:
        """Run full compression pipeline"""
        for idx, step in enumerate(steps):
            print(f"\nRunning {step.step_type} step {idx+1}/{len(steps)}")
            self.run_step(step, idx)
        return self.results
    
    def validate_model_state(self, step_name: str):
        """Validate model weights and outputs after a compression step."""
        print(f"\nValidating model state after {step_name}...")
        
        # Check lm_head weights
        lm_head = self.model.lm_head
        weight = lm_head.weight.data
        bias = lm_head.bias.data if lm_head.bias is not None else None
        
        print("\nLanguage model head statistics:")
        print(f"Weight shape: {weight.shape}")
        print(f"Weight range: [{weight.min():.4f}, {weight.max():.4f}]")
        print(f"Weight mean: {weight.mean():.4f}")
        print(f"NaN in weights: {torch.isnan(weight).any().item()}")
        print(f"Inf in weights: {torch.isinf(weight).any().item()}")
        
        if bias is not None:
            print(f"\nBias range: [{bias.min():.4f}, {bias.max():.4f}]")
            print(f"Bias mean: {bias.mean():.4f}")
            print(f"NaN in bias: {torch.isnan(bias).any().item()}")
            print(f"Inf in bias: {torch.isinf(bias).any().item()}")
        
        # Test forward pass
        sample_input = self.tokenizer(
            "def test():",
            return_tensors="pt",
            truncation=True,
            max_length=32
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**sample_input)
            logits = outputs.logits
            
        print("\nForward pass test:")
        print(f"Output shape: {logits.shape}")
        print(f"Output range: [{logits.min():.4f}, {logits.max():.4f}]")
        print(f"Output mean: {logits.mean():.4f}")
        print(f"NaN in output: {torch.isnan(logits).any().item()}")
        print(f"Inf in output: {torch.isinf(logits).any().item()}")

def create_compression_schedule(
    num_rounds: int = 3,
    initial_prune_ratio: float = 0.2,
    prune_increment: float = 0.1,
    distill_config: Optional[Dict] = None,
    model_config: Optional[Dict] = None
) -> List[CompressionStep]:
    """
    Create a multi-step compression schedule alternating between pruning and distillation.
    
    Args:
        num_rounds: Number of pruning-distillation rounds
        initial_prune_ratio: Initial pruning ratio for heads and neurons (0-1)
        prune_increment: How much to increase pruning ratio each round (0-1)
        distill_config: Base distillation configuration
        model_config: Model architecture config containing num_layers, num_attention_heads, 
                     and intermediate_size (MLP width)
    
    Returns:
        List of CompressionStep configurations
    """
    if model_config is None:
        # Default config for CodeGen-350M
        model_config = {
            "num_layers": 20,
            "num_attention_heads": 16,
            "intermediate_size": 4096
        }

    if distill_config is None:
        distill_config = {
            "num_epochs": 1,
            "learning_rate": 1e-5,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 100,
            "max_length": 256,
            "temperature": 2.0,
            "alpha": 0.5,
            "save_steps": 50,
            "logging_steps": 10,
            "max_grad_norm": 1.0,
            "warmup_ratio": 0.3,
            "weight_decay": 0.01,
        }

    # Calculate total number of heads and neurons
    total_heads = model_config["num_layers"] * model_config["num_attention_heads"]
    total_neurons = model_config["num_layers"] * model_config["intermediate_size"]
    
    steps = []
    
    for round_idx in range(num_rounds):
        # Calculate pruning ratio for this round
        prune_ratio = initial_prune_ratio + (round_idx * prune_increment)
        
        # Calculate number of heads and neurons to prune
        num_heads = int(total_heads * prune_ratio)
        num_neurons = int(total_neurons * prune_ratio)
        
        # Add pruning step
        prune_step = CompressionStep(
            step_type="prune",
            config={
                "prune_spec": {
                    'heads': num_heads,
                    'neurons': num_neurons
                },
                "pruning_args": {
                    "calibration_dataset_size": 1024
                }
            }
        )
        steps.append(prune_step)
        
        # Add distillation step
        distill_step = CompressionStep(
            step_type="distill",
            config=distill_config.copy()
        )
        steps.append(distill_step)
    
    return steps

def filter_simple_problems(dataset):
    """Filter for problems containing specific keywords"""
    return dataset.filter(
        lambda x: any(word in x['text'].lower() 
                     for word in ['reverse', 'sum', 'count', 'average', 'maximum', 'minimum'])
    )

def main():
    """Example usage of compression pipeline with multi-step schedule."""
    from data_loader import load_model, load_dataset
    from evaluators.mbpp_evaluator import MBPPEvaluator
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    # Load model with memory optimizations
    model_name = "Salesforce/codegen-350M-mono"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use fp16
        device_map="cuda",
    )
    
    # Get model config for compression schedule
    model_config = {
        "num_layers": model.config.n_layer,
        "num_attention_heads": model.config.n_head,
        "intermediate_size": model.config.n_embd * 4  # CodeGen uses 4x expansion in MLP
    }
    
    # Load datasets
    train_dataset = load_dataset("mbpp", split="train")
    eval_dataset = load_dataset("mbpp", split="test")
    
    # Filter to simple problems only
    #train_dataset = filter_simple_problems(train_dataset)
    #eval_dataset = filter_simple_problems(eval_dataset)
    
    # Create evaluator with smaller batch size
    evaluator = MBPPEvaluator(
        model=model,
        tokenizer=tokenizer,
        batch_size=16  # Reduced from 8
    )
    
    # Initialize pipeline
    pipeline = CompressionPipeline(
        model=model,
        model_name=model_name,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        evaluator=evaluator,
        experiment_name="codegen350M_prune_and_distill_simple"
    )
    
    # Create compression schedule
    compression_schedule = create_compression_schedule(
        num_rounds=2,  # 2 rounds of pruning-distillation
        initial_prune_ratio=0.2,  # Start by pruning 20%
        prune_increment=0.1,  # Increase pruning by 10% each round
        model_config=model_config,
        distill_config={
            "num_epochs": 4,
            "learning_rate": 1e-5,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 100,
            "max_length": 256,
            "temperature": 2.0,
            "alpha": 0.5,
            "save_steps": 50,
            "logging_steps": 10,
            "max_grad_norm": 1.0,
            "warmup_ratio": 0.3,
            "weight_decay": 0.01,
        }
    )
    
    # Run pipeline
    results = pipeline.run_pipeline(compression_schedule)
    
    # Print final summary
    print("\nExperiment Summary:")
    for idx, result in enumerate(results):
        print(f"\nStep {idx} ({result.step_type}):")
        if 'eval_metrics' in result.metrics:
            eval_metrics = result.metrics['eval_metrics']
            print(f"Pass@1: {eval_metrics['pass@k']['1']:.2f}%")
            print(f"Pass@5: {eval_metrics['pass@k']['5']:.2f}%")
            print(f"Pass@10: {eval_metrics['pass@k']['10']:.2f}%")
    
    print("\nExperiment complete! Results saved in:", pipeline.experiment_dir)

if __name__ == "__main__":
    main() 