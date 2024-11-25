from transformers import AutoModelForCausalLM, GPTNeoConfig, TrainingArguments
from data_loader import load_model, load_dataset
from distillation_trainer import DistillationTrainer
from utils.mbpp_preprocessing import preprocess_mbpp_for_codegen

def create_student_config(teacher_model):
    """
    Create a smaller configuration for the student model.
    Teacher specs:
        - Layers: 20
        - Hidden size: 1024
        - Attention heads: 16
    Student will be ~1/4 the size.
    """
    teacher_config = teacher_model.config
    student_config = GPTNeoConfig(
        vocab_size=teacher_config.vocab_size,
        max_position_embeddings=teacher_config.n_positions,
        hidden_size=768,  # Reduced from 1024
        num_layers=10,    # Reduced from 20
        num_heads=8,      # Reduced from 16
        attention_types=[[["global", "local"], 5]],  # 5 pairs = 10 layers
        window_size=256,
    )
    return student_config

def create_student_model(teacher_model):
    """Create a smaller version of the model for distillation"""
    student_config = create_student_config(teacher_model)
    student_model = AutoModelForCausalLM.from_config(student_config)
    return student_model

def run_distillation(
    model_name: str = "Salesforce/codegen-350M-mono",
    sample_size: int = 100,
    **kwargs
):
    # Load teacher model
    tokenizer, teacher_model = load_model(model_name)
    
    # Create smaller student model
    student_model = create_student_model(teacher_model)
    
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