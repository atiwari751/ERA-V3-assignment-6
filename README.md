# MNIST Digit Classification

![Model Tests](https://github.com/{username}/{repository}/actions/workflows/model_tests.yml/badge.svg)

A PyTorch implementation of a CNN model for MNIST digit classification.

## Model Architecture

The model consists of several convolutional layers followed by fully connected layers. Total parameters are 19,576.

1. **First Conv Block**
   - Conv2d(1, 10, 3)
   - BatchNorm2d(10)
   - ReLU

2. **Second Conv Block**
   - Conv2d(10, 18, 3)
   - ReLU
   - MaxPool2d(2, 2)
   - Dropout(0.15)

3. **Third Conv Block**
   - Conv2d(18, 18, 3)
   - ReLU

4. **Fourth Conv Block**
   - Conv2d(18, 18, 3)
   - BatchNorm2d(18)
   - ReLU
   - MaxPool2d(2, 2)
   - Dropout(0.15)

5. **Fifth Conv Block**
   - Conv2d(18, 18, 3, padding=1)
   - ReLU

6. **Fully Connected Layers**
   - Linear(4*4*18, 30)
   - ReLU
   - Linear(30, 10)

## Hyperparameters

- **Optimizer**: AdamW
  - Learning Rate: 0.001 (initial)
  - Weight Decay: 0.01
  - Betas: (0.9, 0.999)

- **Learning Rate Scheduler**: OneCycleLR
  - Max LR: 0.01
  - Epochs: 20
  - Pct Start: 0.3
  - Div Factor: 10
  - Final Div Factor: 100
  - Anneal Strategy: 'cos'

- **Training**
  - Batch Size: 1024
  - Epochs: 20
  - Loss Function: CrossEntropyLoss with label smoothing 0.1

- **Data Augmentation**
  - Random Rotation: ±10 degrees
  - Random Affine:
    - Translation: ±10%
    - Scale: 90-110%
    - Shear: ±5 degrees

## Results

The model achieves:
- Final Training Accuracy: 98.23%
- Final Test Accuracy: 99.38%
- Best Test Accuracy: 99.40%

## Latest training logs

Loss=1.1327 Batch=58 Accuracy=52.13% LR=0.001606: 100%|████████████████████████████████████████████████████| 59/59 [00:11<00:00,  5.17it/s]

Test set: Average loss: 0.0003, Accuracy: 9335/10000 (93.35%)

Learning Rate: 0.001606
Loss=0.8238 Batch=58 Accuracy=85.83% LR=0.003262: 100%|████████████████████████████████████████████████████| 59/59 [00:11<00:00,  5.19it/s] 

Test set: Average loss: 0.0002, Accuracy: 9719/10000 (97.19%)

Learning Rate: 0.003262
Loss=0.7244 Batch=58 Accuracy=92.50% LR=0.005520: 100%|████████████████████████████████████████████████████| 59/59 [00:11<00:00,  4.99it/s] 

Test set: Average loss: 0.0001, Accuracy: 9830/10000 (98.30%)

Learning Rate: 0.005520
Loss=0.6662 Batch=58 Accuracy=94.74% LR=0.007773: 100%|████████████████████████████████████████████████████| 59/59 [00:12<00:00,  4.85it/s] 

Test set: Average loss: 0.0002, Accuracy: 9838/10000 (98.38%)

Learning Rate: 0.007773
Loss=0.6602 Batch=58 Accuracy=95.75% LR=0.009414: 100%|████████████████████████████████████████████████████| 59/59 [00:12<00:00,  4.65it/s] 

Test set: Average loss: 0.0002, Accuracy: 9815/10000 (98.15%)

Learning Rate: 0.009414
Loss=0.6239 Batch=58 Accuracy=96.42% LR=0.010000: 100%|████████████████████████████████████████████████████| 59/59 [00:12<00:00,  4.67it/s] 

Test set: Average loss: 0.0001, Accuracy: 9879/10000 (98.79%)

Learning Rate: 0.010000
Loss=0.6120 Batch=58 Accuracy=96.81% LR=0.009871: 100%|████████████████████████████████████████████████████| 59/59 [00:13<00:00,  4.53it/s] 

Test set: Average loss: 0.0001, Accuracy: 9898/10000 (98.98%)

Learning Rate: 0.009871
Loss=0.5973 Batch=58 Accuracy=97.13% LR=0.009497: 100%|████████████████████████████████████████████████████| 59/59 [00:12<00:00,  4.78it/s] 

Test set: Average loss: 0.0001, Accuracy: 9915/10000 (99.15%)

Learning Rate: 0.009497
Loss=0.5928 Batch=58 Accuracy=97.22% LR=0.008898: 100%|████████████████████████████████████████████████████| 59/59 [00:11<00:00,  4.95it/s] 

Test set: Average loss: 0.0001, Accuracy: 9895/10000 (98.95%)

Learning Rate: 0.008898
Loss=0.5791 Batch=58 Accuracy=97.48% LR=0.008104: 100%|████████████████████████████████████████████████████| 59/59 [00:12<00:00,  4.73it/s] 

Test set: Average loss: 0.0001, Accuracy: 9926/10000 (99.26%)

Learning Rate: 0.008104
Loss=0.6061 Batch=58 Accuracy=97.70% LR=0.007155: 100%|████████████████████████████████████████████████████| 59/59 [00:11<00:00,  4.99it/s] 

Test set: Average loss: 0.0001, Accuracy: 9906/10000 (99.06%)

Learning Rate: 0.007155
Loss=0.5797 Batch=58 Accuracy=97.72% LR=0.006098: 100%|████████████████████████████████████████████████████| 59/59 [00:12<00:00,  4.79it/s] 

Test set: Average loss: 0.0001, Accuracy: 9932/10000 (99.32%)

Learning Rate: 0.006098
Loss=0.5791 Batch=58 Accuracy=97.85% LR=0.004986: 100%|████████████████████████████████████████████████████| 59/59 [00:12<00:00,  4.90it/s] 

Test set: Average loss: 0.0001, Accuracy: 9925/10000 (99.25%)

Learning Rate: 0.004986
Loss=0.5657 Batch=58 Accuracy=97.83% LR=0.003875: 100%|████████████████████████████████████████████████████| 59/59 [00:11<00:00,  4.95it/s] 

Test set: Average loss: 0.0001, Accuracy: 9937/10000 (99.37%)

Learning Rate: 0.003875
Loss=0.5628 Batch=58 Accuracy=98.00% LR=0.002821: 100%|████████████████████████████████████████████████████| 59/59 [00:13<00:00,  4.53it/s] 

Test set: Average loss: 0.0001, Accuracy: 9928/10000 (99.28%)

Learning Rate: 0.002821
Loss=0.5856 Batch=58 Accuracy=98.13% LR=0.001876: 100%|████████████████████████████████████████████████████| 59/59 [00:12<00:00,  4.65it/s] 

Test set: Average loss: 0.0001, Accuracy: 9937/10000 (99.37%)

Learning Rate: 0.001876
Loss=0.5654 Batch=58 Accuracy=98.24% LR=0.001088: 100%|████████████████████████████████████████████████████| 59/59 [00:12<00:00,  4.73it/s] 

Test set: Average loss: 0.0001, Accuracy: 9940/10000 (99.40%)

Learning Rate: 0.001088
Loss=0.5626 Batch=58 Accuracy=98.20% LR=0.000496: 100%|████████████████████████████████████████████████████| 59/59 [00:12<00:00,  4.83it/s] 

Test set: Average loss: 0.0001, Accuracy: 9940/10000 (99.40%)

Learning Rate: 0.000496
Loss=0.5527 Batch=58 Accuracy=98.15% LR=0.000131: 100%|████████████████████████████████████████████████████| 59/59 [00:11<00:00,  5.03it/s] 

Test set: Average loss: 0.0001, Accuracy: 9939/10000 (99.39%)

Learning Rate: 0.000131
Loss=0.5612 Batch=58 Accuracy=98.23% LR=0.000010: 100%|████████████████████████████████████████████████████| 59/59 [00:12<00:00,  4.84it/s] 

Test set: Average loss: 0.0001, Accuracy: 9938/10000 (99.38%)

Learning Rate: 0.000010