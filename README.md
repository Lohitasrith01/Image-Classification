# Image Classification with VGG-16 and ResNet-18

## Overview
This repository implements and compares the performance of **VGG-16 (Version C)** and **ResNet-18** for image classification tasks. The models are applied to a custom CNN dataset containing images of three classes: **dogs**, **food**, and **vehicles**. The dataset consists of 30,000 images (10,000 images per class), each of size 64x64 pixels.

In this implementation, the goal is to explore the transition from traditional deep CNNs (like VGG-16) to networks with **residual connections** (like ResNet), as well as to apply various techniques for optimizing model performance.

### Dataset
The dataset is composed of 30,000 images, distributed equally across the following classes:
- **Dogs**
- **Food**
- **Vehicles**

Each image is 64x64 pixels in size, making it a moderately small dataset for training and testing image classification models. 

### Key Techniques
1. **VGG-16 (Version C)**: Implementing a deep CNN architecture known for its simple, uniform structure and high performance in image classification tasks.
2. **ResNet-18**: A network architecture that includes **residual connections** to help train very deep networks, mitigating the vanishing gradient problem.
3. **Weight Initialization**: Experimentation with **Xavier** and **He** initialization techniques for optimal model convergence.
4. **Optimizer Comparison**: Performance comparison using **SGD**, **Adam**, and **RMSprop** optimizers.
5. **Regularization Techniques**: Dropout, weight decay (L2 regularization), data augmentation, and early stopping are applied to prevent overfitting and enhance model generalization.

## Key Features
- **Multiple Architectures**: The notebook implements both VGG-16 and ResNet-18 models for comparison.
- **Weight Initialization**: Both **Xavier** and **He** initializations are applied to evaluate their impact on model convergence.
- **Optimizers**: A comparison between **SGD**, **Adam**, and **RMSprop** optimizers helps identify the best optimization approach for image classification.
- **Regularization**: Includes dropout, weight decay, and early stopping techniques to avoid overfitting and improve model robustness.
- **Visualization**: Includes plots of training vs. validation accuracy and loss, confusion matrices, and misclassified images for detailed analysis.

## Setup
### Requirements
- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

Install the required libraries:

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn
```
# Model Architecture Details
## VGG-16 (Version C)
VGG-16 is a relatively simple and deep convolutional architecture that consists of multiple blocks of convolutional layers followed by fully connected layers. The model architecture is well-suited for feature extraction and performs well on image classification tasks. It has a standard architecture of convolution layers, ReLU activations, and max pooling, followed by fully connected layers.

## Key Design Decisions:

- The use of 3x3 convolutions and 2x2 max pooling layers.

- Fully connected layers at the end for classification, with dropout added to prevent overfitting.

## ResNet-18
ResNet-18 is a deep learning model built around residual connections, which allow the gradients to flow more easily through deeper layers. This architecture alleviates the vanishing gradient problem and facilitates training very deep networks.

## Key Design Decisions:

- Residual Blocks: Using skip connections to learn identity mappings, allowing the model to learn residuals instead of direct outputs.

- Global Average Pooling: Instead of flattening the final output, global average pooling is used to reduce overfitting by significantly reducing the number of parameters.

## Model Training
## Weight Initialization:
We experiment with two common weight initialization methods:

- **Xavier Initialization**: Ensures that the weights are set to start from a good starting point.

- **He Initialization**: Specifically designed for ReLU activations, leading to faster convergence.

## Optimizers:
The model is trained using three popular optimizers:

- **SGD (Stochastic Gradient Descent)**: Known for its robustness and general effectiveness in deep learning.

- **Adam (Adaptive Moment Estimation)**: Combines the advantages of both AdaGrad and RMSProp, often yielding faster convergence.

## RMSProp: Efficient for training networks with large datasets.

## Regularization:
- **Dropout**: Used in fully connected layers to prevent overfitting.

- **Weight Decay (L2 Regularization)**: Applied to all weights in the network to further control overfitting.

- **Early Stopping**: Monitors the validation loss and halts training if the model stops improving, thus preventing unnecessary training and overfitting.

## Learning Rate Scheduler:
A learning rate scheduler is used to adjust the learning rate during training. The learning rate is reduced every few epochs to allow the model to settle into a minimum.
## Results
Both models are trained on the ImageNet-like dataset and compared based on training and validation accuracy, loss, and efficiency.


For VGG-16, using the SGD optimizer with batch size 32:

- **Training Accuracy**: ~98.89%

- **Validation Accuracy**: ~93.73%

- **Test Accuracy**: ~96.89%

For ResNet-18, using the Adam optimizer with batch size 64:

- **Training Accuracy**: ~99.62%

- **Validation Accuracy**: ~94.30%

- **Test Accuracy**: ~96.15%



## Conclusion
This project provides a comprehensive comparison of VGG-16 and ResNet-18 for image classification. Through experimentation with different optimizers, weight initializations, and regularization techniques, it demonstrates the effectiveness of residual networks in improving model performance, especially when dealing with deeper architectures. The use of early stopping and data augmentation further ensures that the models generalize well to unseen data.

Future Work
Experiment with more advanced architectures like DenseNet or Inception Networks for improved performance.

Fine-tune hyperparameters such as learning rate, batch size, and the number of layers to optimize performance further.



