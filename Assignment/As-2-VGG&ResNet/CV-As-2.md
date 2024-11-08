---
Create Time: 2nd November 2024
Title: CV-As-2
status: DONE
Author:
  - AllenYGY
tags:
  - Lab
  - CV
  - CNN
  - VGG
  - ResNet
---

# Lab Report: Digit Classification Using Deep Learning Architectures

## Lab Overview

This lab report presents a digit classification task on image data using VGG-16 and ResNet-18 neural networks. The project includes data preprocessing, model training, evaluation, and predictions, leveraging PyTorch for model development.

### Data Preprocessing

The dataset is divided into training, validation, and prediction sets:

- **Training Set**: 7,000 samples for training the model.
- **Validation Set**: 35,000 samples for evaluating model performance.
- **Prediction Set**: 28,000 unlabeled samples for generating final predictions.

The preprocessing steps include resizing the images, normalizing pixel values, and one-hot encoding the labels.

#### Steps:

1. **Load Data**: The data is loaded from CSV files and divided into training and validation sets.
2. **One-Hot Encoding**: Labels are converted to one-hot encoded vectors for multi-class classification.
3. **Normalization and Reshaping**: Pixel values are normalized (divided by 255), and data is reshaped to fit the model’s input size (batch size, channel, height, width).

This preprocessing ensures the data is ready for training the deep learning model, with correctly shaped tensors and normalized inputs for efficient learning.

### VGG-16 Model Architectures

A deep convolutional neural network model with 16 layers. It can be divided into two parts: feature extraction and classification. The feature extraction part consists of convolutional and pooling layers, while the classification part consists of fully connected layers.

#### Feature Extraction and VGG Block

Feature extraction involves stacking 5 VGG blocks.

And Each VGG block consists of multiple convolutional layers followed by batch normalization and ReLU activation.

| Block    | Number of Convolutions | Input Channels | Output Channels | Output Size (H x W) | Max Pooling |
|----------|-------------------------|----------------|-----------------|----------------------|-------------|
| Block 1  | 2                       | 1              | 64              | 112x112              | Yes         |
| Block 2  | 2                       | 64             | 128             | 56x56                | Yes         |
| Block 3  | 3                       | 128            | 256             | 28x28                | Yes         |
| Block 4  | 3                       | 256            | 512             | 14x14                | Yes         |
| Block 5  | 3                       | 512            | 512             | 7x7                  | Yes         |

**Features**:

```python
self.features = nn.Sequential(
	# Block 1: 2Conv + 1MaxPool
    self.vgg_block(num_convs=2, in_channels=1, out_channels=64), # 1*224*224 -> 64*112*112
	# Block 2: 2Conv + 1MaxPool
    self.vgg_block(num_convs=2, in_channels=64, out_channels=128), # 64*112*112 -> 128*56*56
	# Block 3: 3Conv + 1MaxPool
    self.vgg_block(num_convs=3, in_channels=128, out_channels=256), # 128*56*56 -> 256*28*28
    # Block 4: 3Conv + 1MaxPool
    self.vgg_block(num_convs=3, in_channels=256, out_channels=512), # 256*28*28 -> 512*14*14
	# Block 5: 3Conv + 1MaxPool
    self.vgg_block(num_convs=3, in_channels=512, out_channels=512), # 512*14*14 -> 512*7*7
)
```

**VGG Block**:

```python
def vgg_block(self, num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs): # (1 conv + 1 relu) * num_convs
       layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))  # Batch Normalization
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 1 MaxPool
    return nn.Sequential(*layers)
```

#### Classifier

After feature extraction, the **classifier** fully connects the extracted features to output classes. The classifier consists of:

- Two fully connected (linear) layers, each followed by ReLU activation.
- The final fully connected layer maps the outputs to 10 classes (assuming digit classification with 10 categories).

```python
self.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096), # 512*7*7 -> 4096
    nn.ReLU(inplace=True), 
    nn.Linear(4096, 4096),  # 4096 -> 4096
    nn.ReLU(inplace=True), 
    nn.Linear(4096, num_classes),   # 4096 -> 10
)
```

#### Forward Pass

The `forward` method defines how data flows through the network:

- Input passes through the feature extractor (`self.features`) consisting of VGG blocks.
- The output is flattened into a 1D tensor.
- Flattened data is passed through the classifier to produce the final output.

```python
def forward(self, x):
        x = self.features(x)  # 5 VGG blocks
        x = torch.flatten(x, start_dim=1) # flatten
        x = self.classifier(x)  # 3 FC layers
        return x
```

### ResNet-18 Model Architectures

ResNet-18 is a deep neural network architecture designed with **residual blocks** to allow training of very deep networks by addressing the vanishing gradient problem. This model can be divided into three main parts: **preprocessing, residual feature extraction**, and **classification**.

#### Preprocessing

The preprocessing layer is the initial layer of ResNet-18, which prepares the input for deeper layers by reducing its size. This layer consists of:

- A `7x7` convolutional layer with 64 output channels, stride of 2, and padding of 3, followed by batch normalization and ReLU activation.
- A `3x3` max pooling layer with a stride of 2 for downsampling, resulting in reduced spatial dimensions.

**Preprocessing**:

```python
self.preprocess = nn.Sequential(
    nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),  # 1x224x224 -> 64x112x112
    nn.BatchNorm2d(64),  # Batch Normalization
    nn.ReLU(inplace=True),  # ReLU
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 64x112x112 -> 64x56x56
)
```

### Residual Block

The **Residual Block** is the core building block of ResNet. It consists of two `3x3` convolutional layers followed by batch normalization and ReLU activation. Additionally, the block has a **shortcut connection** that adds the input directly to the output of the convolutional layers, allowing the network to "skip" layers, which helps mitigate vanishing gradients.

- **Identity Mapping**: If the number of input and output channels is the same, the input is directly added to the output.
- **1x1 Convolution (Downsampling)**: When there is a change in dimensions or channel depth, a `1x1` convolution is used to match the dimensions before the addition operation.

**Residual Block**:

```python
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        # 3x3 Conv layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 1x1 Conv layer for down-sampling
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False) if use_1x1conv else None

    def forward(self, x):
        identity = x  # Save the input for residual connection
        out = F.relu(self.bn1(self.conv1(x)))  # 3x3 Conv -> BN -> ReLU
        out = self.bn2(self.conv2(out))  # 3x3 Conv -> BN
        # If the number of channels changes or the spatial dimensions change, use 1x1 Conv for down-sampling
        if self.conv3:
            identity = self.conv3(x)
        out += identity
        return F.relu(out)  # Apply ReLU activation at the end
```

#### Feature Extraction

The feature extraction part is composed of **four groups of residual blocks**:

- Each group has a specific number of output channels (64, 128, 256, 512).
- The stride of the first block in each group is set to downsample the spatial dimensions, while the subsequent blocks in the group maintain the spatial dimensions.
  
**Code for Residual Block Groups**:
```python
self.features = nn.Sequential(
    self.residual_block(64, 64, 2, stride=1),  # 64x56x56 -> 64x56x56
    self.residual_block(64, 128, 2, stride=2),  # 64x56x56 -> 128x28x28
    self.residual_block(128, 256, 2, stride=2),  # 128x28x28 -> 256x14x14
    self.residual_block(256, 512, 2, stride=2),  # 256x14x14 -> 512x7x7
)
```

#### Classifier

The classifier consists of:

- An **adaptive average pooling layer**, which reduces the spatial dimensions to `1x1`.
- A **fully connected (linear) layer** that maps the final features to the number of classes (10 in this case, assuming a digit classification task).

**Classifier**:

```python
self.classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d((1, 1)),  # Adaptive average pooling layer
    nn.Flatten(),  # Flatten
    nn.Linear(512, num_classes)  # Fully connected layer
)
```

#### Forward Pass

The `forward` method defines the data flow through the ResNet model:

- Input passes through the preprocessing layer.
- The result is then passed through the residual blocks.
- The output is pooled, flattened, and passed through the classifier to obtain the final predictions.

**Forward Pass Code**:
```python
def forward(self, x):
    x = self.preprocess(x)
    x = self.features(x)
    x = self.classifier(x)          
    return x
```

### Model Training

The models were trained using the Adam optimizer and a categorical cross-entropy loss function. Training involved feeding augmented data batches through the network, optimizing weights iteratively.

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(num_epochs):
    train(train_loader, model, loss_fn, optimizer)
```

### Model Evaluation

After training, the models were evaluated using accuracy, precision, recall, and F1-score metrics to assess their performance on validation data.

|   Network      | Accuracy | Precision | Recall | F1 Score |
|---------|----------|-----------|--------|----------|
| VGG-16  | 0.9651   | 0.9662    | 0.9651 | 0.9651   |
| ResNet-18 | 0.9637 | 0.966     | 0.9637 | 0.9639   |

Both VGG-16 and ResNet-18 performed well on the digit classification task, achieving accuracy above 96%. The key differences lie in training speed and computational efficiency:

- **VGG-16**: Achieved slightly higher accuracy (96.51%) but required longer training time due to its high parameter count, especially in fully connected layers. This makes it effective for tasks with moderate complexity, though it has a larger memory footprint.

- **ResNet-18**: Trained faster due to its residual connections, which improve gradient flow and reduce the risk of vanishing gradients. Its architecture, with fewer parameters, is more memory-efficient, making it suitable for deeper networks or larger datasets.

In summary, **VGG-16** is advantageous when simplicity and interpretability are key, while **ResNet-18** offers efficiency and scalability, especially useful in resource-constrained environments or for more complex tasks.

### Training Results Visualization

![Visualization](https://cdn.jsdelivr.net/gh/AllenYGY/ImageSpace@main/uPic/00ftRr.png)

## Comparison of VGG-16 and ResNet-18

| Feature                    | VGG-16                                     | ResNet-18                                        |
|----------------------------|--------------------------------------------|--------------------------------------------------|
| **Architecture Depth**     | 16 layers                                  | 18 layers                                        |
| **Core Building Block**    | Convolutional layers with max pooling      | Residual blocks with identity mapping            |
| **Feature Extraction**     | Stacks of VGG blocks                       | Stacks of residual blocks                        |
| **Parameter Count**        | High (138 million parameters)              | Relatively lower (11 million parameters)         |
| **Computational Complexity** | Higher due to more parameters and fully connected layers | Lower due to fewer parameters and optimized residual structure |
| **Gradient Flow**          | Standard feed-forward, prone to vanishing gradient | Residual connections improve gradient flow, reducing vanishing gradient issues |
| **Training Time**          | Slower due to high parameter count         | Faster training, as fewer parameters and optimized architecture |
| **Memory Usage**           | High, due to large fully connected layers  | Lower, optimized for efficiency with fewer layers |
| **Weight Initialization**  | Requires careful initialization to prevent vanishing gradients | More stable training with residual connections, requires less careful initialization |
| **Performance on Simple Tasks** | Very effective for smaller datasets or simpler tasks | Slightly overpowered for simple tasks but maintains performance |
| **Performance on Complex Tasks** | Limited scalability in very deep versions due to vanishing gradients | Scales effectively to deeper versions like ResNet-50, ResNet-101 |
| **Use Cases**              | Image classification on medium complexity tasks | Versatile, effective for both simple and complex tasks |
| **Overall Strengths**      | Simplicity, effective on moderate tasks    | Efficient gradient flow, good scalability for deep networks |
| **Drawbacks**              | High computational and memory cost         | Performance gain in simpler versions (ResNet-18) not always substantial |

## Weight initialization

### Xavier Initialization

Xavier Initialization is designed to maintain the variance of activations across layers. It works well with layers that use **sigmoid** or **tanh** activations.

**Formula**:
$$
W \sim \mathcal{U} \left(-\sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}, \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}\right)
$$
where $\text{fan\_in}$ is the number of input units, and $\text{fan\_out}$ is the number of output units for the layer.

**Usage**: Applied to models where activation functions like tanh or sigmoid are used. It helps maintain the activation variance, avoiding issues with vanishing or exploding gradients.

### Kaiming Initialization

Kaiming Initialization is tailored for layers with **ReLU** activations. It adjusts for the fact that ReLU outputs are zero for half the inputs, which can cause gradients to vanish if not handled properly.

**Formula**:

$$
W \sim \mathcal{N} \left(0, \frac{2}{\text{fan\_in}}\right)
$$
where $\text{fan\_in}$ is the number of input units.

**Usage**: Commonly used in CNNs with ReLU layers, such as VGG and ResNet models. It preserves the variance of gradients, which helps maintain effective backpropagation in deep networks.

| Initialization       | Formula                                             | Best Used With                  | Benefits                                |
|----------------------|-----------------------------------------------------|---------------------------------|-----------------------------------------|
| **Xavier**           | $W \sim \mathcal{U} \left(-\sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}, \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}\right)$ | Sigmoid, Tanh activations       | Maintains activation variance, stabilizes gradients |
| **Kaiming**          | $W \sim \mathcal{N} \left(0, \frac{2}{\text{fan\_in}}\right)$ | ReLU activations               | Helps prevent vanishing gradients in ReLU networks |

- **VGG-16** and **ResNet-18** both benefit from **Kaiming Initialization** due to their reliance on ReLU activations.
- **Xavier Initialization** is an alternative when experiment with different activation functions, such as tanh.

These initializations ensure effective gradient flow, improve convergence, and reduce the risk of vanishing or exploding gradients.

## Normalization methods

### Batch Normalization (BN)
Batch Normalization normalizes inputs across a mini-batch, reducing internal covariate shift and helping to stabilize training. It is applied after convolutional layers in both VGG and ResNet models, accelerating convergence and improving gradient flow.

**Formula**:

$$
\hat{x}_i = \frac{x_i - \mu_\text{batch}}{\sqrt{\sigma_\text{batch}^2 + \epsilon}}
$$

**Benefits**: Reduces sensitivity to initialization, improves regularization, and stabilizes gradients.

### Layer Normalization (LN)
Layer Normalization normalizes inputs across all features for each sample, independent of batch size. It’s ideal for RNNs and small batch sizes.

**Formula**:

$$
\hat{x}_i = \frac{x_i - \mu_\text{layer}}{\sqrt{\sigma_\text{layer}^2 + \epsilon}}
$$

**Benefits**: Suitable for tasks with variable-length sequences, especially useful in RNNs and NLP models.

### Instance Normalization (IN)
Instance Normalization normalizes each sample in a batch independently, typically used in tasks like style transfer where each sample has unique features.

**Formula**:

$$
\hat{x}_{ij} = \frac{x_{ij} - \mu_\text{instance}}{\sqrt{\sigma_\text{instance}^2 + \epsilon}}
$$
**Benefits**: Retains unique sample characteristics, less reliant on batch statistics.

### Group Normalization (GN)
Group Normalization splits channels into groups and normalizes each group independently, useful for small-batch training.

**Formula**:

$$
\hat{x}_{ij} = \frac{x_{ij} - \mu_\text{group}}{\sqrt{\sigma_\text{group}^2 + \epsilon}}
$$

**Benefits**: Reduces reliance on large batches, ideal for segmentation and detection tasks.

| Technique           | Application                      | Benefits                                    |
|---------------------|----------------------------------|---------------------------------------------|
| **Batch Norm**      | Across mini-batches              | Faster convergence, stabilizes gradients    |
| **Layer Norm**      | Within each layer                | Less batch-dependent, good for RNNs         |
| **Instance Norm**   | Per-sample normalization         | Preserves unique features, ideal for style transfer |
| **Group Norm**      | Divides channels into groups     | Consistent with small batches               |

## Conclusion

This experiment successfully implemented and evaluated VGG-16 and ResNet-18 architectures for handwritten digit classification. Through data preprocessing, model training, and evaluation, the results demonstrated that both models effectively classify digit images with high accuracy. Batch Normalization, Kaiming Initialization, and efficient use of ReLU activations contributed to stable training and strong model performance.

While both models achieved similar accuracy, ResNet-18 showed faster training due to its optimized residual blocks, which improve gradient flow in deeper networks. Further improvements could be achieved by experimenting with different normalization techniques (e.g., Layer, Instance, and Group Normalization) and weight initializations to optimize convergence further. Additionally, exploring hyperparameter tuning, advanced data augmentation, and other architectures, such as deeper ResNet variants, could enhance classification accuracy and robustness for more complex tasks.
