import pandas as pd
import sys


def split_dataframe(df, pos_a_train, pos_b_train, neg_a_train, neg_b_train, pos_a_test1, pos_b_test1, neg_a_test1, neg_b_test1,
                    pos_a_test2, pos_b_test2, neg_a_test2, neg_b_test2):
    # Filter rows by group and clinical label
    pos_a = df[(df['clinical_label'] == 1) & (df['Exposure parameter group'] == 'A')]
    pos_b = df[(df['clinical_label'] == 1) & (df['Exposure parameter group'] == 'B')]
    neg_a = df[(df['clinical_label'] == 0) & (df['Exposure parameter group'] == 'A')]
    neg_b = df[(df['clinical_label'] == 0) & (df['Exposure parameter group'] == 'B')]

    # Split data into train set
    train_set = pd.concat([
        pos_a.sample(n=pos_a_train, random_state=42),
        pos_b.sample(n=pos_b_train, random_state=42),
        neg_a.sample(n=neg_a_train, random_state=42),
        neg_b.sample(n=neg_b_train, random_state=42)
    ])

    # Split data into test set 1
    test_set_1 = pd.concat([
        pos_a.sample(n=pos_a_test1, random_state=42),
        pos_b.sample(n=pos_b_test1, random_state=42),
        neg_a.sample(n=neg_a_test1, random_state=42),
        neg_b.sample(n=neg_b_test1, random_state=42)
    ])

    # Split data into test set 2
    test_set_2 = pd.concat([
        pos_a.sample(n=pos_a_test2, random_state=42),
        pos_b.sample(n=pos_b_test2, random_state=42),
        neg_a.sample(n=neg_a_test2, random_state=42),
        neg_b.sample(n=neg_b_test2, random_state=42)
    ])

    return train_set, test_set_1, test_set_2

def main():
    # Get command line arguments
    argv = sys.argv
    input_csv = argv[1]  # Input CSV file
    output_train_csv = argv[2]  # Output train set CSV file
    output_test1_csv = argv[3]  # Output test set 1 CSV file
    output_test2_csv = argv[4]  # Output test set 2 CSV file

    # Specified numbers for each subset
    pos_a_train = int(argv[5])
    pos_b_train = int(argv[6])
    neg_a_train = int(argv[7])
    neg_b_train = int(argv[8])
    pos_a_test1 = int(argv[9])
    pos_b_test1 = int(argv[10])
    neg_a_test1 = int(argv[11])
    neg_b_test1 = int(argv[12])
    pos_a_test2 = int(argv[13])
    pos_b_test2 = int(argv[14])
    neg_a_test2 = int(argv[15])
    neg_b_test2 = int(argv[16])

    # Read the input DataFrame
    df = pd.read_csv(input_csv)

    # Split the DataFrame
    train_set, test_set_1, test_set_2 = split_dataframe(
        df, pos_a_train, pos_b_train, neg_a_train, neg_b_train, pos_a_test1, pos_b_test1, neg_a_test1, neg_b_test1,
        pos_a_test2, pos_b_test2, neg_a_test2, neg_b_test2
    )

    # Save the output DataFrames to CSV files
    train_set.to_csv(output_train_csv, index=False)
    test_set_1.to_csv(output_test1_csv, index=False)
    test_set_2.to_csv(output_test2_csv, index=False)

if __name__ == "__main__":
    main()
