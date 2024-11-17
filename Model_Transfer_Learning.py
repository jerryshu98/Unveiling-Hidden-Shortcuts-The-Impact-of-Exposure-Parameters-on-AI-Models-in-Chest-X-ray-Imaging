import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import sys

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

# Main function
def main():
    # Get command line arguments
    argv = sys.argv
    train_csv_path = argv[1]  # Train CSV file path
    val_csv_path = argv[2]  # Validation CSV file path
    test_csv_path = argv[3]  # Test CSV file path
    pkl_path = argv[4]  # Pickle file with DICOM IDs and images
    label_column = argv[5]  # Column name for labels (e.g., 'exposure parameter group')
    model_path = argv[6]  # Path to the model to be loaded
    save_model_path = argv[7]  # Path to save the new model

    # Load train, validation, and test data
    X_train, y_train = load_data(train_csv_path, pkl_path, label_column)
    X_val, y_val = load_data(val_csv_path, pkl_path, label_column)
    X_test, y_test = load_data(test_csv_path, pkl_path, label_column)

    # Load the pre-trained model
    model = load_model(model_path)

    # Get the target layer name
    target_layer_name = model.layers[-3].name
    layer_name = target_layer_name

    # Extract the intermediate layer output
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output_train = intermediate_layer_model.predict(X_train)
    intermediate_output_val = intermediate_layer_model.predict(X_val)

    # Create a new input layer for the intermediate output
    input_layer = tf.keras.Input(shape=intermediate_output_train.shape[1:])

    # Copy the layers from the original model after the target layer
    x = input_layer
    for layer in model.layers[model.layers.index(model.get_layer(layer_name)) + 1:]:
        x = layer(x)

    # Create the new model
    new_model = Model(inputs=input_layer, outputs=x)
    new_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auroc', curve='ROC'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    # Train the new model using the intermediate outputs
    new_model.fit(intermediate_output_train, y_train, epochs=10, validation_data=(intermediate_output_val, y_val), batch_size=16)

    # Save the new model
    new_model.save(save_model_path, save_format='tf')

if __name__ == "__main__":
    main()
