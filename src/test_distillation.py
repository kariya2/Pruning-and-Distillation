from transformers import TrainingArguments
from data_loader import load_model, load_dataset
from distillation_trainer import DistillationTrainer

def test_distillation():
    # Load models
    model_name = "Salesforce/codegen-350M-mono"
    tokenizer, teacher_model = load_model(model_name)
    _, student_model = load_model(model_name)
    
    # Load dataset
    dataset = load_dataset("mbpp", split="train", sample_size=100)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir="./distillation_test",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        learning_rate=1e-4,
        logging_steps=10,
    )
    
    # Initialize trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        model=student_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        temperature=2.0,
        alpha=0.5
    )
    
    # Train
    trainer.train()

if __name__ == "__main__":
    test_distillation() 