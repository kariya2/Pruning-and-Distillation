from transformers import Trainer, TrainingArguments
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union

def validate_model_state(model, tokenizer):
        """Validate model weights and outputs after a compression step."""

        print("We HERE")
        
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
        print(f"NaN in output: {torch.isnan(logits).any().item()}")
        print(f"Inf in output: {torch.isinf(logits).any().item()}")

class DistillationTrainer(Trainer):
    """Extends HuggingFace Trainer for knowledge distillation"""
    
    def __init__(self, teacher_model, temperature=2.0, alpha=0.5, **kwargs):
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        super().__init__(**kwargs)
        
    def _compute_kl_loss(self, student_logits, teacher_logits):
        """Compute KL divergence loss with numerical stability checks"""
        eps = 1e-7
        
        # Scale logits before softmax but don't multiply final loss by T^2
        s_logits = student_logits / self.temperature 
        t_logits = teacher_logits / self.temperature
        
        # Add epsilon for numerical stability
        s_logits = torch.clamp(s_logits, min=-100, max=100)  # Add clipping
        t_logits = torch.clamp(t_logits, min=-100, max=100)  # Add clipping
        
        # Compute log softmax and softmax with dimension checks
        log_softmax_student = F.log_softmax(s_logits, dim=-1)
        softmax_teacher = F.softmax(t_logits, dim=-1)
        
        # Check for invalid values
        if torch.isnan(log_softmax_student).any() or torch.isnan(softmax_teacher).any():
            print("Warning: NaN values detected in logits")
            log_softmax_student = torch.nan_to_num(log_softmax_student, nan=0.0)
            softmax_teacher = torch.nan_to_num(softmax_teacher, nan=1.0/log_softmax_student.size(-1))
        
        # Compute KL divergence without the temperature scaling
        loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
        return loss_fct(log_softmax_student, softmax_teacher)
    
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """Compute combined distillation and task loss with detailed logging"""
        # Get student outputs
        #validate_model_state(model, self.tokenizer)
        outputs = model(**inputs)
        student_logits = outputs.logits
        
        # Check for NaNs immediately after forward pass
        if torch.isnan(student_logits).any():
            print("\nNaN detected in student logits immediately after forward pass!")
            print(f"Input shape: {inputs['input_ids'].shape}")
            print(f"Student logits shape: {student_logits.shape}")
            
            # Check each layer's output
            print("\nChecking intermediate layer outputs:")
            with torch.no_grad():
                hidden_states = model.transformer(inputs['input_ids']).last_hidden_state
                print(f"Transformer output shape: {hidden_states.shape}")
                print(f"Transformer output has NaNs: {torch.isnan(hidden_states).any().item()}")
                
                # Check lm_head weights
                print("\nChecking lm_head weights:")
                print(f"Weight has NaNs: {torch.isnan(model.lm_head.weight).any().item()}")
                if model.lm_head.bias is not None:
                    print(f"Bias has NaNs: {torch.isnan(model.lm_head.bias).any().item()}")
            
            raise ValueError("NaN values detected in model outputs - stopping training")
        
        # Get teacher outputs
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
            
            # Log teacher logits state
            if torch.isnan(teacher_logits).any():
                print("\nNaN detected in teacher logits:")
                print(f"- NaN percentage: {torch.isnan(teacher_logits).float().mean() * 100:.2f}%")
        
        # Compute distillation loss
        distillation_loss = self._compute_kl_loss(student_logits, teacher_logits)
        
        # Compute task loss if labels available
        if "labels" in inputs:
            task_loss = outputs.loss
            
            # Log task loss state
            if torch.isnan(task_loss).any():
                print("\nNaN detected in task loss")
                task_loss = torch.zeros_like(task_loss)
            
            loss = self.alpha * task_loss + (1 - self.alpha) * distillation_loss
        else:
            loss = distillation_loss
        
        return (loss, outputs) if return_outputs else loss