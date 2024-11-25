from datasets import Dataset
from transformers import PreTrainedTokenizer
from typing import Union, Dict, Any, List
import torch

def prepare_tokenizer(tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
    """
    Ensures tokenizer has proper padding configuration.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # Need to resize embeddings for the new token
        if hasattr(tokenizer, 'model_max_length'):
            tokenizer.model_max_length = min(tokenizer.model_max_length, 2048)
    return tokenizer

def prepare_mbpp_prompt(text: str) -> str:
    """
    Creates a standardized MBPP prompt for CodeGen models.
    
    Args:
        text: Problem description from MBPP dataset
    
    Returns:
        Formatted prompt string
    """
    return f"""# Write a Python function for this task. Only the function, no tests or examples:
# {text}

def solution"""

def tokenize_for_codegen(
    prompts: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    padding_side: str = 'left'
) -> Dict[str, torch.Tensor]:
    """
    Tokenizes prompts consistently for CodeGen models.
    
    Args:
        prompts: List of formatted prompts
        tokenizer: CodeGen tokenizer
        max_length: Maximum sequence length
        padding_side: Which side to apply padding ('left' or 'right')
    
    Returns:
        Tokenized inputs ready for model
    """
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        padding_side=padding_side
    )
    
    return inputs

def preprocess_mbpp_for_codegen(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512
) -> Dataset:
    """
    Preprocesses MBPP dataset for CodeGen models using causal language modeling format.
    """
    tokenizer = prepare_tokenizer(tokenizer)
    
    def preprocess_function(examples: Dict[str, List]) -> Dict[str, Any]:
        # First tokenize prompts to get their lengths
        prompts = [prepare_mbpp_prompt(text) for text in examples['text']]
        prompt_encodings = tokenizer(prompts, add_special_tokens=False)
        prompt_lengths = [len(ids) for ids in prompt_encodings['input_ids']]
        
        # Tokenize full sequences in one go
        full_texts = [f"{prompt}{code}" for prompt, code in zip(prompts, examples['code'])]
        model_inputs = tokenizer(
            full_texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Create labels tensor and mask prompts efficiently
        labels = model_inputs['input_ids'].clone()
        for idx, length in enumerate(prompt_lengths):
            labels[idx, :length] = -100
            
        model_inputs['labels'] = labels
        return model_inputs
    
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return processed_dataset 