import pandas as pd
import sys
import itertools
import numpy as np

def get_thresholds(df, column):
    # Calculate 10%, 20%, ..., 100% percentiles for each column
    return [np.percentile(df[column], i) for i in range(10, 101, 10)]

def find_best_split(df, clinical_label):
    best_split = None
    max_difference = -float('inf')
    columns = ['xraytubecurrent', 'exposure', 'exposuretime', 'exposureinmus']

    # Get thresholds for all exposure parameters
    thresholds = {col: get_thresholds(df, col) for col in columns}

    # Iterate over all possible three-node combinations
    for root_col, left_col, right_col in itertools.permutations(columns, 3):
        for root_thresh in thresholds[root_col]:
            group_a_root = df[df[root_col] <= root_thresh]
            group_b_root = df[df[root_col] > root_thresh]

            for left_thresh in thresholds[left_col]:
                group_a_left = group_a_root[group_a_root[left_col] <= left_thresh]
                group_b_left = group_a_root[group_a_root[left_col] > left_thresh]

                for right_thresh in thresholds[right_col]:
                    group_a_right = group_b_root[group_b_root[right_col] <= right_thresh]
                    group_b_right = group_b_root[group_b_root[right_col] > right_thresh]

                    # Calculate the ratio of clinical label positive and negative in group A
                    positive_ratio_a = (
                        group_a_left[group_a_left[clinical_label] == 1].shape[0] +
                        group_a_right[group_a_right[clinical_label] == 1].shape[0]
                    ) / df[df[clinical_label] == 1].shape[0]

                    negative_ratio_a = (
                        group_a_left[group_a_left[clinical_label] == 0].shape[0] +
                        group_a_right[group_a_right[clinical_label] == 0].shape[0]
                    ) / df[df[clinical_label] == 0].shape[0]

                    # Calculate the difference in ratios
                    difference = abs(positive_ratio_a - negative_ratio_a)

                    # Update the best split point
                    if difference > max_difference:
                        max_difference = difference
                        best_split = (root_col, root_thresh, left_col, left_thresh, right_col, right_thresh)

    return best_split

def process_dataframe(df, clinical_label):
    # Find the best split point
    root_col, root_thresh, left_col, left_thresh, right_col, right_thresh = find_best_split(df, clinical_label)

    # Assign groups based on the best split point
    def assign_group(row):
        if row[root_col] <= root_thresh:
            if row[left_col] <= left_thresh:
                return 'A'
            else:
                return 'B'
        else:
            if row[right_col] <= right_thresh:
                return 'A'
            else:
                return 'B'

    df['Exposure parameter group'] = df.apply(assign_group, axis=1)

    # Output the best split parameters
    print(f"Root node: {root_col} with threshold {root_thresh}")
    print(f"Left node: {left_col} with threshold {left_thresh}")
    print(f"Right node: {right_col} with threshold {right_thresh}")

    return df

def main():
    # Get command line arguments
    argv1 = sys.argv[1]  # Clinical label file path or name
    argv2 = sys.argv[2]  # Input DataFrame CSV file
    argv3 = sys.argv[3]  # Output processed DataFrame CSV file

    # Read clinical label
    clinical_label = argv1  # Assuming argv1 is the clinical label name, adjust as needed

    # Read the input DataFrame
    df = pd.read_csv(argv2)

    # Process the DataFrame
    processed_df = process_dataframe(df, clinical_label)

    # Save the processed DataFrame to the output file
    processed_df.to_csv(argv3, index=False)

if __name__ == "__main__":
    main()
