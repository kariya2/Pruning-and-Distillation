from transformers import TrainingArguments
from data_loader import load_model, load_dataset
from distillation_trainer import DistillationTrainer
from utils.mbpp_preprocessing import preprocess_mbpp_for_codegen

def run_distillation(
    model_name: str = "Salesforce/codegen-350M-mono",
    sample_size: int = 100,
    **kwargs
):
    # Load models
    tokenizer, teacher_model = load_model(model_name)
    _, student_model = load_model(model_name)
    
    # Load and preprocess dataset
    raw_dataset = load_dataset("mbpp", split="train", sample_size=sample_size)
    processed_dataset = preprocess_mbpp_for_codegen(raw_dataset, tokenizer)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir="./distillation_output",
        num_train_epochs=kwargs.get('num_epochs', 2),
        per_device_train_batch_size=kwargs.get('batch_size', 4),
        learning_rate=kwargs.get('learning_rate', 1e-4),
        logging_steps=10,
    )
    
    # Initialize trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        model=student_model,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=tokenizer,
        temperature=kwargs.get('temperature', 2.0),
        alpha=kwargs.get('alpha', 0.5)
    )
    
    # Train
    return trainer.train() 