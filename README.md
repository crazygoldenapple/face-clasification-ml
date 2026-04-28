Image Classification: Anime / Cartoon / Human

Project Overview
This project implements a 3-class image classifier designed to distinguish between Anime, Cartoon, and Human images.

The defining characteristic of this project is the construction of a Neural Network from scratch using only NumPy. No high-level deep learning frameworks like Keras or PyTorch were used for the model implementation, ensuring a deep understanding of backpropagation and optimization logic.

Pipeline Structure
The project is organized into six distinct parts:

Data Loading and Visualization: Loading images from a structured directory and performing preprocessing.

Feature Engineering: Implementation of Histogram of Oriented Gradients (HOG), color histograms, and raw pixel extraction.

ANN Implementation: Building the neural network from scratch with a sanity check.

Baseline Model Training: Training using all three feature types concatenated.

Test-set Evaluation: Performance measurement on unseen data.

Hyperparameter Optimization: Grid search, k-fold cross-validation, and feature combination experiments.

Technical Configuration
The project utilizes the following default settings for image processing and model architecture:

Image Size: Resized to 64×64 pixels.

Data Split: 80% Training / 20% Testing (Stratified).

Default ANN Architecture: Two hidden layers (256 and 128 units respectively).

Optimization: Learning rate of 0.01, Batch size of 32, and 50 training epochs.

Data Preprocessing
Images are expected to be stored in the following folder structure:

Plaintext
Data/
  Anime/
  Cartoon/
  Human/
During loading, every image is converted to RGB, resized to 64×64, and normalized to a pixel range of [0,1].

Requirements
To run this project, the following libraries are required:

numpy (Core math and model implementation)

matplotlib & seaborn (Visualization)

PIL (Image processing)

scikit-image (HOG and color features)

scikit-learn (Data splitting and metrics)

tqdm (Progress tracking)

Results Summary
The best configuration found during experimentation achieved an F1-score of 0.9961.

Best Feature Method: Combined HOG and Pixels.

Best Hyperparameters: Single hidden layer (128), learning rate 0.01, batch size 16.

AI Tools Statement
Claude AI (Anthropic) was utilized to assist with code structure, feature extraction implementation logic, visualization design, and documentation.
