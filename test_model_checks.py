import pytest
import torch
import torch.nn as nn
from model import Net

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def test_parameter_count(capsys):
    model = Net()
    total_params = count_parameters(model)
    print(f"\nTotal parameters in model: {total_params:,}")
    print(f"Parameter budget: 20,000")
    assert total_params < 20000, f"Model has {total_params:,} parameters, exceeding limit of 20,000"

def test_batch_norm_presence(capsys):
    model = Net()
    bn_layers = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    print(f"\nBatch Normalization layers found: {len(bn_layers)}")
    print(f"BatchNorm locations: {[name for name, module in model.named_modules() if isinstance(module, nn.BatchNorm2d)]}")
    assert len(bn_layers) > 0, "Model should use Batch Normalization"

def test_dropout_presence(capsys):
    model = Net()
    dropout_layers = [module for module in model.modules() if isinstance(module, nn.Dropout2d)]
    print(f"\nDropout layers found: {len(dropout_layers)}")
    print(f"Dropout rates: {[layer.p for layer in dropout_layers]}")
    assert len(dropout_layers) > 0, "Model should use Dropout"

def test_fc_presence(capsys):
    model = Net()
    fc_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    print(f"\nFully Connected layers found: {len(fc_layers)}")
    print(f"FC layer dimensions: {[(fc.in_features, fc.out_features) for fc in fc_layers]}")
    assert len(fc_layers) > 0, "Model should use either Fully Connected layer or GAP"

def test_model_structure(capsys):
    model = Net()
    print("\nDetailed Model Structure:")
    for name, module in model.named_children():
        print(f"{name}: {module.__class__.__name__}")
        if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            print(f"    Dimensions: {module.in_features} → {module.out_features}")
        elif hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
            print(f"    Channels: {module.in_channels} → {module.out_channels}") 