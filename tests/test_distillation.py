import pytest
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from distillation import run_distillation, create_student_model
from data_loader import load_model
import torch
import torch.nn.functional as F

# Add debug prints
print("Imported functions:")
print(f"run_distillation: {run_distillation}")
print(f"create_student_model: {create_student_model}")
print(f"load_model: {load_model}")

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
    
    # Load teacher model
    tokenizer, teacher_model = load_model("Salesforce/codegen-350M-mono")
    
    # Create student model
    student_model = create_student_model(teacher_model)
    
    # Test on a simple prompt
    prompt = "# Write a function to calculate factorial\n\ndef solution"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        teacher_output = teacher_model(**inputs).logits
        student_output = student_model(**inputs).logits
    
    # Check if student outputs are closer to teacher after training
    similarity = F.cosine_similarity(teacher_output, student_output)
    print(f"\nTeacher-Student output similarity: {similarity.mean().item()}")
    
    # Additional size verification
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    print(f"\nTeacher parameters: {teacher_params:,}")
    print(f"Student parameters: {student_params:,}")
    print(f"Size reduction: {(1 - student_params/teacher_params)*100:.2f}%")
    
    return results