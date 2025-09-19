# Covid-Detection-Using-X--Ray-
# COVID-19 Detection from Chest X-Ray Images

This repository contains code for building, training, evaluating, and deploying deep learning models to classify chest X-ray images as COVID-19 positive or Normal. The project includes data preprocessing, model definition, training with data augmentation, evaluation, visualization, and single image prediction using transfer learning.

---

## Repository Structure

| File Name                         | Description                                                        |
|----------------------------------|--------------------------------------------------------------------|
| `file_listing_script.py`          | Script to list all files in the dataset directory.                 |
| `visualize_dataset_images.py`     | Loads and displays sample images from Normal and COVID classes.    |
| `cnn_model_definition.py`         | Defines a CNN architecture for binary classification.              |
| `train_model_with_augmentation.py`| Data augmentation and training script for the CNN model.           |
| `plot_training_history.py`        | Plots training and validation accuracy and loss over epochs.       |
| `evaluate_model_predictions.py`   | Prints final training/validation accuracy and maps predictions.    |
| `evaluate_model_metrics.py`       | Generates confusion matrix and classification report visualization.|
| `mobilenetv2_single_image_predictor.py` | Transfer learning with MobileNetV2 and single image prediction.   |
| `README.md`                      | Project documentation.                                             |

---

## Requirements

Make sure to install the required Python packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
**##Usage Overview
1. List Dataset Files

Run file_listing_script.py to verify your dataset directory structure and files.

2. Visualize Sample Images

Use visualize_dataset_images.py to load and display example images from both Normal and COVID categories.

3. Define and Build Model

The CNN model is defined in cnn_model_definition.py. This can be customized or replaced with transfer learning models.

4. Train the Model

Train the CNN model with data augmentation by running train_model_with_augmentation.py. This script uses ImageDataGenerator for preprocessing.

5. Visualize Training History

After training, plot accuracy and loss graphs using plot_training_history.py to evaluate model learning.

6. Evaluate Model Predictions

evaluate_model_predictions.py shows predictions on the validation dataset and prints accuracy scores.

7. Generate Evaluation Metrics

Create confusion matrix heatmaps and classification reports with evaluate_model_metrics.py to better understand performance.

8. Transfer Learning and Single Image Prediction

Use the MobileNetV2 based model in mobilenetv2_single_image_predictor.py for improved accuracy and run single image predictions.
