from transformers import Trainer, TrainingArguments, get_cosine_schedule_with_warmup
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
        
        # Define layer-wise clipping thresholds
        self.layer_clip_thresholds = {
            'wte': 5.0,  # Embedding layer
            'h.0': 5.0,  # Early layers
            'h.1': 5.0,
            'h.2': 5.0,
            'h.3': 5.0,
            'h.4': 4.5,
            'h.5': 4.5,
            'h.6': 4.0,  # Middle layers
            'h.7': 4.0,
            'h.8': 3.5,
            'h.9': 3.5,
            'h.10': 3.0,
            'h.11': 3.0,
            'h.12': 2.5,  # Later layers
            'h.13': 2.5,
            'h.14': 2.0,
            'h.15': 2.0,
            'h.16': 1.5,
            'h.17': 1.5,
            'h.18': 1.0,  # Final layers
            'h.19': 1.0,
            'lm_head': 1.0  # Output layer
        }
        
        # Disable global gradient clipping since we're doing layer-wise
        if 'max_grad_norm' in kwargs.get('args', {}).__dict__:
            print("Warning: Disabling global gradient clipping in favor of layer-wise clipping")
            kwargs['args'].max_grad_norm = None
        
        super().__init__(**kwargs)
    
    def _clip_layer_gradients(self):
        """Apply layer-wise gradient clipping"""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Extract layer number for transformer layers
                threshold = 5.0  # default threshold
                
                if 'lm_head' in name:
                    threshold = self.layer_clip_thresholds['lm_head']
                elif 'transformer.h.' in name:
                    # Extract layer number
                    layer_num = int(name.split('transformer.h.')[1].split('.')[0])
                    layer_key = f'h.{layer_num}'
                    if layer_key in self.layer_clip_thresholds:
                        threshold = self.layer_clip_thresholds[layer_key]
                elif 'wte' in name:
                    threshold = self.layer_clip_thresholds['wte']
                
                # Compute gradient norm
                grad_norm = param.grad.norm()
                
                # Clip if necessary
                if grad_norm > threshold:
                    param.grad.data.mul_(threshold / (grad_norm + 1e-6))
                    #print(f"\nClipped {name} gradient:")
                    #print(f"  Before: {grad_norm:.4f}")
                    #print(f"  After: {threshold:.4f}")
                    #print(f"  Layer: {f'h.{layer_num}' if 'transformer.h.' in name else ('lm_head' if 'lm_head' in name else 'other')}")
                    #print(f"  Threshold: {threshold}")
    
    def _compute_kl_loss(self, student_logits, teacher_logits):
        """Compute KL divergence loss with enhanced numerical stability."""
        # Print initial ranges
        #print(f"\nInitial ranges:")
        #print(f"Student: [{student_logits.min():.4f}, {student_logits.max():.4f}], mean: {student_logits.mean():.4f}")
        #print(f"Teacher: [{teacher_logits.min():.4f}, {teacher_logits.max():.4f}], mean: {teacher_logits.mean():.4f}")

        # Scale down BEFORE any other operations
        scale = 0.1
        student_logits = student_logits * scale
        teacher_logits = teacher_logits * scale

        #print(f"\nAfter scaling (x{scale}):")
        #print(f"Student: [{student_logits.min():.4f}, {student_logits.max():.4f}], mean: {student_logits.mean():.4f}")
        #print(f"Teacher: [{teacher_logits.min():.4f}, {teacher_logits.max():.4f}], mean: {teacher_logits.mean():.4f}")

        # Subtract max for numerical stability
        student_max = student_logits.max(dim=-1, keepdim=True)[0].detach()
        teacher_max = teacher_logits.max(dim=-1, keepdim=True)[0].detach()

        student_logits = student_logits - student_max
        teacher_logits = teacher_logits - teacher_max

        #print(f"\nAfter max subtraction:")
        #print(f"Student: [{student_logits.min():.4f}, {student_logits.max():.4f}], mean: {student_logits.mean():.4f}")
        #print(f"Teacher: [{teacher_logits.min():.4f}, {teacher_logits.max():.4f}], mean: {teacher_logits.mean():.4f}")

        # Apply temperature
        s_logits = student_logits / self.temperature
        t_logits = teacher_logits / self.temperature

        # Compute log probabilities
        log_softmax_student = F.log_softmax(s_logits, dim=-1)
        log_softmax_teacher = F.log_softmax(t_logits, dim=-1)

        #print(f"\nAfter log softmax:")
        #print(f"Student log range: [{log_softmax_student.min():.4f}, {log_softmax_student.max():.4f}]")
        #print(f"Teacher log range: [{log_softmax_teacher.min():.4f}, {log_softmax_teacher.max():.4f}]")

        # Compute KL divergence with log_target=True
        loss = F.kl_div(
            log_softmax_student,
            log_softmax_teacher,
            reduction="batchmean",
            log_target=True
        )
        print(f"\nFinal loss: {loss.item():.4f}")

        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN/Inf in KL loss, falling back to scaled MSE")
            return 0.1 * F.mse_loss(student_logits, teacher_logits)

        return loss
    
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """Only compute loss, no gradient operations"""
        outputs = model(**inputs, output_hidden_states=True)
        student_logits = outputs.logits
        
        # Check for NaNs in weights first
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN weights in {name}")
                raise RuntimeError("Training failed - NaN values detected in model weights")
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        
        distillation_loss = self._compute_kl_loss(student_logits, teacher_logits)
        task_loss = outputs.loss if "labels" in inputs else None
        
        loss = distillation_loss if task_loss is None else (
            self.alpha * task_loss + (1 - self.alpha) * distillation_loss
        )
        
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Handle all gradient operations in one place"""
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Check initial state
        #self.nan_debugger.step()
        #if self.nan_debugger.nan_found:
            #self._save_debug_state("pre_forward")
            #raise RuntimeError("NaN detected before forward pass")
        
        # Loss computation and backward
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch)
        
        # Check after forward pass
        ##self.nan_debugger.step()
        ##if self.nan_debugger.nan_found:
            ##self._save_debug_state("post_forward")
            #raise RuntimeError("NaN detected after forward pass")
        
        loss.backward()
        
        # Check after backward
        #self.nan_debugger.step()
        #if self.nan_debugger.nan_found:
            #self._save_debug_state("post_backward")
            #raise RuntimeError("NaN detected after backward pass")
        
        # Clip gradients
        self._clip_layer_gradients()
        
        # Check after gradient clipping
        #self.nan_debugger.step()
        #if self.nan_debugger.nan_found:
            #self._save_debug_state("post_clip")
            #raise RuntimeError("NaN detected after gradient clipping")
        
        # Optimizer step
        self.optimizer.step()
        
        # Check after optimizer step
        #self.nan_debugger.step()
        #if self.nan_debugger.nan_found:
            #self._save_debug_state("post_optimizer")
            #raise RuntimeError("NaN detected after optimizer step")
        
        self.lr_scheduler.step()
        model.zero_grad()
        
        return loss.detach()
    
    def _save_debug_state(self, stage):
        """Save model and optimizer state when NaN is detected"""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'teacher_state': self.teacher_model.state_dict(),
            'step': self.nan_debugger.step_count,
            'stage': stage,
            'last_loss': self.state.log_history[-1] if self.state.log_history else None,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }, f'nan_debug_{stage}.pt')
    
    def create_optimizer(self):
        """Override to create AdamW optimizer with better settings."""
        decay_parameters = self.get_decay_parameter_names(self.model)
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if n in decay_parameters and p.requires_grad],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if n not in decay_parameters and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

        from torch.optim import AdamW
        optimizer_cls = AdamW
        
        # Increase base learning rate
        initial_lr = 5e-5  # Increased from 1e-5
        
        optimizer_kwargs = {
            "lr": initial_lr,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        }

        self.optimizer = optimizer_cls(
            optimizer_grouped_parameters,
            **optimizer_kwargs
        )

        # Adjust warmup and schedule
        num_training_steps = self.args.num_train_epochs * len(self.train_dataset) // (self.args.train_batch_size * self.args.gradient_accumulation_steps)
        num_warmup_steps = min(1000, num_training_steps // 10)  # Use fixed warmup steps instead of ratio

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        return self.optimizer