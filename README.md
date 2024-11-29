# MNIST Digit Classification

A PyTorch implementation of a CNN model for MNIST digit classification.

## Model Architecture

The model consists of several convolutional layers followed by fully connected layers:

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

## Latest Training Logs 