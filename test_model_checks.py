import pytest
import torch
import torch.nn as nn
from model import Net

def test_parameter_count():
    model = Net()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters in model: {total_params}")
    assert total_params < 20000, f"Model has {total_params} parameters, should be less than 20000"

def test_batch_norm_presence():
    model = Net()
    has_batch_norm = any(isinstance(module, nn.BatchNorm2d) for module in model.modules())
    print(f"\nBatch Normalization present: {has_batch_norm}")
    assert has_batch_norm, "Model should use Batch Normalization"

def test_dropout_presence():
    model = Net()
    has_dropout = any(isinstance(module, nn.Dropout2d) for module in model.modules())
    print(f"\nDropout present: {has_dropout}")
    assert has_dropout, "Model should use Dropout"

def test_fc_presence():
    model = Net()
    has_fc = any(isinstance(module, nn.Linear) for module in model.modules())
    print(f"\nFully Connected layer present: {has_fc}")
    assert has_fc, "Model should use either Fully Connected layer or GAP"

def test_model_structure():
    model = Net()
    print("\nModel Structure:")
    for name, module in model.named_children():
        print(f"{name}: {module.__class__.__name__}") 