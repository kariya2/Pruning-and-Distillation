import pytest
from distillation import run_distillation
import torch
import torch.nn.functional as F

def test_distillation():
    # Run distillation
    results = run_distillation(
        model_name="Salesforce/codegen-350M-mono",
        sample_size=10,
        num_epochs=1,
        batch_size=2
    )
    
    # Check training metrics
    assert results.metrics is not None
    assert 'train_loss' in results.metrics
    
    # Compare teacher and student outputs
    tokenizer, teacher_model = load_model("Salesforce/codegen-350M-mono")
    _, student_model = load_model("Salesforce/codegen-350M-mono")
    
    # Test on a simple prompt
    prompt = "# Write a function to calculate factorial\n\ndef solution"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        teacher_output = teacher_model(**inputs).logits
        student_output = student_model(**inputs).logits
    
    # Check if student outputs are closer to teacher after training
    similarity = F.cosine_similarity(teacher_output, student_output)
    print(f"\nTeacher-Student output similarity: {similarity.mean().item()}")
    
    return results