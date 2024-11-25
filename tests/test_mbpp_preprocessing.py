import pytest
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from utils.mbpp_preprocessing import (
    prepare_mbpp_prompt,
    tokenize_for_codegen,
    preprocess_mbpp_for_codegen,
    prepare_tokenizer
)

@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")

@pytest.fixture
def sample_dataset():
    return Dataset.from_dict({
        'text': [
            'write a function to check if a number is prime',
            'write a function to reverse a string'
        ],
        'code': [
            'def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0: return False\n    return True',
            'def reverse_string(s):\n    return s[::-1]'
        ],
        'test_list': [
            ['assert is_prime(17) == True'],
            ['assert reverse_string("hello") == "olleh"']
        ],
        'task_id': [1, 2],
        'test_setup_code': ['', ''],
        'challenge_test_list': [
            ['assert is_prime(29) == True'],
            ['assert reverse_string("world") == "dlrow"']
        ]
    })

def test_prepare_mbpp_prompt():
    text = "write a function to check if a number is prime"
    expected = """# Write a Python function for this task. Only the function, no tests or examples:
# write a function to check if a number is prime

def solution"""
    assert prepare_mbpp_prompt(text) == expected

def test_tokenize_for_codegen(tokenizer):
    # Prepare tokenizer first
    tokenizer = prepare_tokenizer(tokenizer)
    
    prompts = [
        "# Test prompt 1\ndef solution",
        "# Test prompt 2\ndef solution"
    ]
    result = tokenize_for_codegen(prompts, tokenizer)
    
    assert 'input_ids' in result
    assert 'attention_mask' in result
    assert result['input_ids'].shape[0] == 2  # batch size
    assert torch.all(result['attention_mask'].bool() == (result['input_ids'] != tokenizer.pad_token_id))

def test_preprocess_mbpp_for_codegen(tokenizer, sample_dataset):
    processed = preprocess_mbpp_for_codegen(sample_dataset, tokenizer)
    
    # Check that the processed dataset has only the needed columns
    expected_columns = {'input_ids', 'attention_mask', 'labels'}
    assert set(processed.column_names) == expected_columns
    
    # Get first example
    example = processed[0]
    
    # Check tensor shapes match
    input_shape = len(example['input_ids'])
    label_shape = len(example['labels'])
    attention_shape = len(example['attention_mask'])
    
    assert input_shape == label_shape, f"Input shape {input_shape} != Label shape {label_shape}"
    assert input_shape == attention_shape, f"Input shape {input_shape} != Attention shape {attention_shape}"
    
    # Check prompt masking in labels
    prompt = prepare_mbpp_prompt(sample_dataset[0]['text'])
    prompt_tokens = tokenizer(prompt, return_tensors="pt")['input_ids'][0]
    prompt_length = len(prompt_tokens)
    
    # Verify prompt tokens are masked with -100 in labels
    assert all(label == -100 for label in example['labels'][:prompt_length]), \
        "Prompt tokens should be masked with -100 in labels"
    
    # Verify code tokens are not masked in labels
    assert any(label != -100 for label in example['labels'][prompt_length:]), \
        "Code tokens should not be masked in labels"
    
    # Verify input_ids contain both prompt and code
    decoded_input = tokenizer.decode(example['input_ids'])
    assert "Write a Python function" in decoded_input, "Prompt not found in input_ids"
    assert "is_prime" in decoded_input, "Code not found in input_ids"

def test_processed_dataset_format(tokenizer, sample_dataset):
    """Test that processed dataset matches what DistillationTrainer expects"""
    processed = preprocess_mbpp_for_codegen(sample_dataset, tokenizer)
    
    # Print detailed info about the processed dataset
    print("\nProcessed dataset info:")
    print(f"Columns: {processed.column_names}")
    print(f"Number of examples: {len(processed)}")
    
    # Print first example in detail
    example = processed[0]
    print("\nFirst example:")
    for key, value in example.items():
        if isinstance(value, list):
            print(f"{key}: type=list, length={len(value)}")
        else:
            print(f"{key}: type={type(value)}")
    
    # Verify list properties
    assert isinstance(example['input_ids'], list)
    assert isinstance(example['attention_mask'], list)
    assert isinstance(example['labels'], list)
    
    return processed  # Return for manual inspection if needed