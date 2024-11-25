from transformers import Trainer, TrainingArguments
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union

class DistillationTrainer(Trainer):
    """Extends HuggingFace Trainer for knowledge distillation"""
    
    def __init__(self, teacher_model, temperature=2.0, alpha=0.5, **kwargs):
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        super().__init__(**kwargs)
        
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Compute distillation loss combining teacher and student outputs.
        
        Args:
            model: The student model
            inputs: The inputs to the model
            num_items_in_batch: Number of items in batch (added to match parent signature)
            return_outputs: Whether to return model outputs along with the loss
        """
        # Get student outputs
        outputs = model(**inputs)
        student_logits = outputs.logits
        
        # Get teacher outputs
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
            
        # Compute distillation loss
        loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
        distillation_loss = (
            loss_fct(
                torch.nn.functional.log_softmax(student_logits / self.temperature, dim=-1),
                torch.nn.functional.softmax(teacher_logits / self.temperature, dim=-1)
            )
            * (self.temperature ** 2)
        )
        
        # Compute standard loss
        if "labels" in inputs:
            labels = inputs["labels"]
            standard_loss = outputs.loss
            # Combined loss
            loss = self.alpha * standard_loss + (1 - self.alpha) * distillation_loss
        else:
            loss = distillation_loss
        
        return (loss, outputs) if return_outputs else loss