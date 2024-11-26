import pytest
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.activation_pruning import ActivationBasedPruning

@pytest.fixture
def tiny_model():
    """Returns a tiny CodeGen model for testing"""
    model = AutoModelForCausalLM.from_pretrained(
        "Salesforce/codegen-350M-mono",  # Using smaller model for testing
        torch_dtype='auto',
        device_map='auto'
    )
    return model

@pytest.fixture
def tiny_tokenizer():
    """Returns a CodeGen tokenizer for testing"""
    return AutoTokenizer.from_pretrained(
        "Salesforce/codegen-350M-mono",
    )

@pytest.fixture
def tiny_dataset():
    """Returns a small sample of MBPP dataset for testing"""
    dataset = load_dataset("mbpp", split="test").select(range(10))  # Small subset for testing
    return dataset

@pytest.fixture
def setup_pruning(tiny_model, tiny_tokenizer, tiny_dataset):
    """Creates an ActivationBasedPruning instance with minimal test components"""
    pruner = ActivationBasedPruning(
        model=tiny_model,
        tokenizer=tiny_tokenizer,
        dataset=tiny_dataset,
        calibration_dataset_size=4
    )
    return pruner 