import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import sys

# Define function to build the model
def build_model(input_shape=(256, 256, 1), num_classes=1):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = models.Sequential([
        layers.experimental.preprocessing.Resizing(224, 224),
        layers.Conv2D(3, (3, 3), padding='same'),  # Convert grayscale to RGB
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='sigmoid')  # Binary classification
    ])

    return model

# Load DataFrame and images
def load_data(df_path, pkl_path, label_column):
    # Load dataframe from CSV
    df = pd.read_csv(df_path)
    
    # Load image data from pickle file
    with open(pkl_path, 'rb') as file:
        image_data = pickle.load(file)
    
    # Extract matching image arrays for dicoms in DataFrame
    images = []
    labels = []
    for dicom_id in df['dicom']:
        if dicom_id in image_data:
            images.append(image_data[dicom_id])
            labels.append(df[df['dicom'] == dicom_id][label_column].values[0])
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

# Balance dataset by downsampling Positive-A and Positive-B, Negative-A and Negative-B
def balance_data(df):
    # Separate classes
    pos_a = df[(df['clinical_label'] == 1) & (df['Exposure parameter group'] == 'A')]
    pos_b = df[(df['clinical_label'] == 1) & (df['Exposure parameter group'] == 'B')]
    neg_a = df[(df['clinical_label'] == 0) & (df['Exposure parameter group'] == 'A')]
    neg_b = df[(df['clinical_label'] == 0) & (df['Exposure parameter group'] == 'B')]

    # Downsample to match the smaller class size
    pos_min = min(len(pos_a), len(pos_b))
    neg_min = min(len(neg_a), len(neg_b))

    pos_a_downsampled = resample(pos_a, replace=False, n_samples=pos_min, random_state=42)
    pos_b_downsampled = resample(pos_b, replace=False, n_samples=pos_min, random_state=42)
    neg_a_downsampled = resample(neg_a, replace=False, n_samples=neg_min, random_state=42)
    neg_b_downsampled = resample(neg_b, replace=False, n_samples=neg_min, random_state=42)

    # Combine the downsampled datasets
    balanced_df = pd.concat([pos_a_downsampled, pos_b_downsampled, neg_a_downsampled, neg_b_downsampled])
    
    return balanced_df

# Main function
def main():
    # Get command line arguments
    argv = sys.argv
    df_train_path = argv[1]  # Train CSV file from Grouping_Algorithm.py
    df_test1_path = argv[2]  # Test set 1 CSV file
    df_test2_path = argv[3]  # Test set 2 CSV file
    pkl_path = argv[4]  # Pickle file with DICOM IDs and images
    model_output_path = argv[5]  # Output path for saving the model
    label_column = argv[6]  # Column name for labels (e.g., 'clinical_label' or 'Exposure parameter group')
    balance_data_flag = argv[7].lower() == 'true'  # Flag to indicate if balancing is needed

    # Load train and validation data
    df_train = pd.read_csv(df_train_path)
    if balance_data_flag:
        df_train = balance_data(df_train)
    df_train.to_csv("balanced_train_data.csv", index=False)  # Save the balanced data for reference

    X_train, y_train = load_data("balanced_train_data.csv", pkl_path, label_column)
    X_val, y_val = load_data(df_test1_path, pkl_path, label_column)
    X_test2, y_test2 = load_data(df_test2_path, pkl_path, label_column)

    # Build the model
    model = build_model(input_shape=(256, 256, 1))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auroc', curve='ROC')])

    # Define a callback to save the best model based on validation loss
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_output_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    # Train the model
    model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), batch_size=16, callbacks=[checkpoint_callback])

    # Load the best model
    best_model = tf.keras.models.load_model(model_output_path)

    # Evaluate the model on Test set 1 and Test set 2
    test1_results = best_model.evaluate(X_val, y_val, verbose=0)
    test2_results = best_model.evaluate(X_test2, y_test2, verbose=0)

    # Print AUROC for Test set 1 and Test set 2
    print(f"Test Set 1 - AUROC: {test1_results[2]:.4f}")
    print(f"Test Set 2 - AUROC: {test2_results[2]:.4f}")

if __name__ == "__main__":
    main()
