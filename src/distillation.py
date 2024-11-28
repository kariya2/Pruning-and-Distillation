from transformers import TrainingArguments
from data_loader import load_model, load_dataset
from distillation_trainer import DistillationTrainer
from utils.mbpp_preprocessing import preprocess_mbpp_for_codegen

def run_distillation(
    teacher_model,
    student_model,
    tokenizer,
    dataset=None,
    sample_size: int = 100,
    **kwargs
):
    """
    Run knowledge distillation from teacher to student model.
    
    Args:
        teacher_model: The teacher model (previous stable model)
        student_model: The student model (pruned model to be trained)
        tokenizer: The tokenizer to use
        dataset: Optional dataset (if not provided, will load MBPP)
        sample_size: Number of samples to use if loading dataset
        **kwargs: Additional training arguments
    """
    # Load dataset if not provided
    if dataset is None:
        dataset = load_dataset("mbpp", split="train", sample_size=sample_size)
    
    # Preprocess dataset
    processed_dataset = preprocess_mbpp_for_codegen(dataset, tokenizer)
    
    # Setup training arguments with all kwargs
    training_args = TrainingArguments(
        output_dir="./distillation_output",
        num_train_epochs=kwargs.get('num_epochs', 2),
        per_device_train_batch_size=kwargs.get('batch_size', 4),
        learning_rate=kwargs.get('learning_rate', 1e-4),
        logging_steps=kwargs.get('logging_steps', 10),
        save_strategy="no",
        report_to="none",
        max_grad_norm=kwargs.get('max_grad_norm', 1.0),
        gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 1),
        warmup_steps=kwargs.get('warmup_steps', 0),
        warmup_ratio=kwargs.get('warmup_ratio', 0.0),
        weight_decay=kwargs.get('weight_decay', 0.0),
        save_steps=kwargs.get('save_steps', 500),
    )
    
    # Initialize trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        model=student_model,  # student model is the main model
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=tokenizer,
        temperature=kwargs.get('temperature', 2.0),
        alpha=kwargs.get('alpha', 0.5)
    )
    
    # Train and return results
    return trainer.train() 