
# Unified Gender and Emotion Classification Using Convolutional Neural Networks

## Overview
This project aims to develop a unified machine learning model capable of predicting both the gender of individuals and their emotional state from images. By leveraging deep learning techniques, specifically Convolutional Neural Networks (CNNs), the model aims to accurately classify images into gender categories (male/female) and emotional states (happy/sad). The dual classification model has potential applications in various fields, including facial recognition, social robotics, and sentiment analysis.

## Objective
To create a robust and accurate CNN-based model that can predict both gender and emotion from images, addressing the challenges of imbalanced datasets and integrating multi-task learning for comprehensive image classification.

## Steps Involved in the Project

### 1. Data Collection and Preparation
- Acquire and load datasets containing images labeled for gender and emotion.
- Preprocess the datasets, including normalization and reshaping labels.

### 2. Model Development
- Define a CNN model architecture suitable for image classification.
- Develop separate CNN models for emotion detection and combined gender-emotion classification.

### 3. Training and Validation
- Train the emotion detection model using labeled emotion data.
- Use early stopping and model checkpoints to optimize training.
- Evaluate the model on test data and visualize training history.

### 4. Dual Classification Model
- Integrate emotion detection into the gender classification process.
- Create new labels combining gender and emotion.
- Train the combined gender-emotion model using the integrated labels.
- Apply class balancing techniques like class weighting and undersampling to address class imbalances.

### 5. Model Evaluation
- Evaluate all trained models on a separate test dataset.
- Compare the performance of models trained with different techniques (simple training, early stopping, class weights, undersampling).
- Plot training history (accuracy and loss) for each model to analyze performance over epochs.

### 6. Testing on New Images
- Test the final models on a new set of images.
- Decode the predicted labels and visualize the results.

## Conclusion
The project successfully developed and trained a unified model capable of predicting both gender and emotional state from images. By integrating emotion detection into the gender classification process, the model provides a comprehensive understanding of the individual's characteristics depicted in images. The use of advanced CNN architectures, along with techniques like early stopping, model checkpointing, class weighting, and undersampling, contributed to the model's accuracy and robustness.

However, while the accuracy of all four models in predicting emotions is commendable, there is a slight shortfall in gender prediction, particularly for female images. This performance gap is primarily due to the dataset being heavily skewed towards males, with the emotion dataset lacking female images entirely. To address these issues and improve the model's performance, several enhancements can be implemented:
1. **Changing Model Architecture**: Experimenting with different and more complex CNN architectures could help the model learn better features for gender prediction.
2. **Bigger and Balanced Dataset**: Acquiring a larger and more balanced dataset that includes a wide spectrum of images with diverse emotions and genders can significantly improve the model's accuracy and generalization.
3. **Data Augmentation**: Implementing data augmentation techniques to artificially increase the diversity of the dataset could also help in mitigating the skewness of the dataset.

By incorporating these improvements, the model can achieve higher accuracy and reliability in predicting both gender and emotional states from images, making it more robust and applicable to real-world scenarios.
