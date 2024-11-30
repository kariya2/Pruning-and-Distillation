from transformers import TrainingArguments
from data_loader import load_model, load_dataset
from distillation_trainer import DistillationTrainer
from utils.mbpp_preprocessing import preprocess_mbpp_for_codegen
import torch

def validate_model(model, tokenizer, model_name: str = "model"):
    """Validate model weights and outputs before distillation."""
    print(f"\nValidating {model_name} before distillation:")
    
    # Check all transformer layer weights
    print("\nTransformer layer statistics:")
    for i, layer in enumerate(model.transformer.h):
        # Check attention weights
        qkv = layer.attn.qkv_proj.weight
        out = layer.attn.out_proj.weight
        
        print(f"\nLayer {i} attention:")
        print(f"QKV weight: range [{qkv.min():.4f}, {qkv.max():.4f}], "
              f"mean {qkv.mean():.4f}, std {qkv.std():.4f}")
        print(f"Out weight: range [{out.min():.4f}, {out.max():.4f}], "
              f"mean {out.mean():.4f}, std {out.std():.4f}")
        
        # Check MLP weights
        fc_in = layer.mlp.fc_in.weight
        fc_out = layer.mlp.fc_out.weight
        
        print(f"MLP:")
        print(f"FC in:  range [{fc_in.min():.4f}, {fc_in.max():.4f}], "
              f"mean {fc_in.mean():.4f}, std {fc_in.std():.4f}")
        print(f"FC out: range [{fc_out.min():.4f}, {fc_out.max():.4f}], "
              f"mean {fc_out.mean():.4f}, std {fc_out.std():.4f}")
    
    # Check final LM head
    lm_head = model.lm_head
    weight = lm_head.weight.data
    bias = lm_head.bias.data if lm_head.bias is not None else None
    
    print("\nLanguage model head statistics:")
    print(f"Weight shape: {weight.shape}")
    print(f"Weight range: [{weight.min():.4f}, {weight.max():.4f}]")
    print(f"Weight mean: {weight.mean():.4f}")
    print(f"Weight std: {weight.std():.4f}")
    print(f"NaN in weights: {torch.isnan(weight).any().item()}")
    print(f"Inf in weights: {torch.isinf(weight).any().item()}")
    
    if bias is not None:
        print(f"\nBias range: [{bias.min():.4f}, {bias.max():.4f}]")
        print(f"Bias mean: {bias.mean():.4f}")
        print(f"Bias std: {bias.std():.4f}")
        print(f"NaN in bias: {torch.isnan(bias).any().item()}")
        print(f"Inf in bias: {torch.isinf(bias).any().item()}")
    
    # Test forward pass
    sample_input = tokenizer(
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
    print(f"Output std: {logits.std():.4f}")
    print(f"NaN in output: {torch.isnan(logits).any().item()}")
    print(f"Inf in output: {torch.isinf(logits).any().item()}")

def run_distillation(
    teacher_model,
    student_model,
    tokenizer,
    dataset=None,
    sample_size: int = 100,
    **kwargs
):
    """Run knowledge distillation from teacher to student model."""
    
    if dataset is None:
        dataset = load_dataset("mbpp", split="train", sample_size=sample_size)
    
    processed_dataset = preprocess_mbpp_for_codegen(dataset, tokenizer)
    
    # Print actual config being used
    print("\nDistillation Configuration:")
    for k, v in kwargs.items():
        print(f"- {k}: {v}")
    
    training_args = TrainingArguments(
        output_dir="./distillation_output",
        num_train_epochs=kwargs.get('num_epochs', 1),
        per_device_train_batch_size=kwargs.get('batch_size', 4),
        learning_rate=kwargs.get('learning_rate', 1e-6),
        logging_steps=kwargs.get('logging_steps', 10),
        save_strategy="no",
        report_to="none",
        max_grad_norm=kwargs.get('max_grad_norm', 1.0),
        gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 4),
        warmup_steps=kwargs.get('warmup_steps', 100),
        warmup_ratio=kwargs.get('warmup_ratio', 0.3),
        weight_decay=kwargs.get('weight_decay', 0.01),
        save_steps=kwargs.get('save_steps', 500),
    )
    
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        model=student_model,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=tokenizer,
        temperature=kwargs.get('temperature', 1.0),
        alpha=kwargs.get('alpha', 0.5)
    )
    
    # Create optimizer first
    trainer.create_optimizer()
    
    # Now print optimizer config
    print("\nOptimizer Configuration:")
    print(f"Learning rate: {trainer.optimizer.param_groups[0]['lr']}")
    print(f"Beta1 (momentum): {trainer.optimizer.param_groups[0]['betas'][0]}")
    print(f"Beta2: {trainer.optimizer.param_groups[0]['betas'][1]}")
    
    return trainer.train() 