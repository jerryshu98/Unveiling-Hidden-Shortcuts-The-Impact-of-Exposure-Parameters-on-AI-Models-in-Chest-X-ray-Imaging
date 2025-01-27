# Overview
This repository contains the code used in the study titled "Unveiling Hidden Shortcuts: The Impact of Exposure Parameters on AI Models in Chest X-ray Imaging." The repository includes four scripts for processing, splitting, training, and transfer learning of medical imaging data. These scripts implement the experimental components described in the paper, focusing on how exposure parameters influence AI model predictions and lead to potential biases.

The scripts are:
1. `Grouping_Algorithm.py`: Groups chest X-ray images based on exposure parameters to create Group A and Group B.
2. `Dataset_Selection.py`: Splits the dataset into different training and testing sets to simulate various experimental conditions, such as biased models.
3. `Model_Training.py`: Trains models for different tasks (e.g., pneumothorax detection, race classification) using bias, balance, origin, or exposure-based settings.
4. `Model_Transfer_Learning.py`: Applies transfer learning on models generated by `Model_Training.py` to analyze their retained exposure parameter information.

## Grouping_Algorithm.py
This script categorizes chest X-ray images into Group A and Group B based on four exposure parameters: ExposureTime, Exposure, XRayTubeCurrent, and ExposureInuAs. The grouping algorithm aims to maximize the difference between positive and negative labels in each group.

### Input/Output
- **Input**: Takes three command line arguments:
  1. `clinical_label`: Name of the column representing clinical labels.
  2. `input_csv`: Path to the input CSV file with medical imaging data.
  3. `output_csv`: Path to save the output CSV file with group assignment results.

### Usage
```sh
python Grouping_Algorithm.py <clinical_label> <input_csv> <output_csv>
```

### Example
```sh
python Grouping_Algorithm.py clinical_label input_data.csv output_grouped_data.csv
```

## Dataset_Selection.py
This script splits the grouped dataset produced by `Grouping_Algorithm.py` into three different datasets (train set, test set 1, and test set 2) for creating scenarios like biased models, balanced models, and more.

### Input/Output
- **Input**: Takes 16 command line arguments:
  1. `input_csv`: Path to the CSV file produced by `Grouping_Algorithm.py`.
  2. `output_train_csv`: Path to save the train set CSV.
  3. `output_test1_csv`: Path to save test set 1 CSV.
  4. `output_test2_csv`: Path to save test set 2 CSV.
  5-16. Counts for each group in the three sets:
     - `pos_a_train`, `pos_b_train`, `neg_a_train`, `neg_b_train`: Counts for the train set.
     - `pos_a_test1`, `pos_b_test1`, `neg_a_test1`, `neg_b_test1`: Counts for test set 1.
     - `pos_a_test2`, `pos_b_test2`, `neg_a_test2`, `neg_b_test2`: Counts for test set 2.

### Usage
```sh
python Dataset_Selection.py <input_csv> <output_train_csv> <output_test1_csv> <output_test2_csv> \
<pos_a_train> <pos_b_train> <neg_a_train> <neg_b_train> \
<pos_a_test1> <pos_b_test1> <neg_a_test1> <neg_b_test1> \
<pos_a_test2> <pos_b_test2> <neg_a_test2> <neg_b_test2>
```

### Example
```sh
python Dataset_Selection.py output_grouped_data.csv train_set.csv test1_set.csv test2_set.csv \
50 50 50 50 20 20 20 20 10 10 10 10
```

## Model_Training.py
This script trains deep learning models for different tasks (e.g., pneumothorax detection, race classification). The models can be trained using different settings such as biased, balanced, origin, or exposure models based on the provided arguments.

### Input/Output
- **Input**: Takes 8 command line arguments:
  1. `df_train_path`: Path to the train set CSV file.
  2. `df_test1_path`: Path to the test set 1 CSV file.
  3. `df_test2_path`: Path to the test set 2 CSV file.
  4. `pkl_path`: Path to the pickle file containing DICOM IDs and corresponding images.
  5. `model_output_path`: Path to save the trained model.
  6. `label_column`: Name of the column representing labels (e.g., 'clinical_label' or 'Exposure parameter group').
  7. `balance_data_flag`: Boolean flag to indicate whether to balance the data (e.g., 'true' or 'false').

### Usage
```sh
python Model_Training.py <df_train_path> <df_test1_path> <df_test2_path> <pkl_path> <model_output_path> <label_column> <balance_data_flag>
```

### Example
```sh
python Model_Training.py train_set.csv test1_set.csv test2_set.csv image_data.pkl resnet50_model clinical_label true
```

## Model_Transfer_Learning.py
This script applies transfer learning on the models generated by `Model_Training.py`. It extracts features from an intermediate layer of a pre-trained model and retrains it to analyze the extent of retained information from exposure parameters.

### Input/Output
- **Input**: Takes 8 command line arguments:
  1. `train_csv_path`: Path to the train set CSV file.
  2. `val_csv_path`: Path to the validation set CSV file.
  3. `test_csv_path`: Path to the test set CSV file.
  4. `pkl_path`: Path to the pickle file containing DICOM IDs and corresponding images.
  5. `label_column`: Name of the column representing labels (e.g., 'clinical_label').
  6. `model_path`: Path to the pre-trained model to be loaded.
  7. `save_model_path`: Path to save the newly trained model.

### Usage
```sh
python Model_Transfer_Learning.py <train_csv_path> <val_csv_path> <test_csv_path> <pkl_path> <label_column> <model_path> <save_model_path>
```

### Example
```sh
python Model_Transfer_Learning.py train_set.csv val_set.csv test_set.csv image_data.pkl clinical_label pretrained_model.keras new_transfer_model
```

# Requirements
- Python 3.x
- pandas
- numpy
- TensorFlow
- scikit-learn

# Notes
- The scripts are used to analyze and quantify the influence of exposure parameters on model predictions. Each script implements a part of the experimental workflow described in the paper to examine biases introduced by exposure parameters in chest X-ray imaging.

