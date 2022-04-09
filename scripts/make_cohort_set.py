import argparse
import pandas as pd
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_list_path', type=str, default="data/train.csv")
    parser.add_argument('--cohort_save_path', type=str, default="data/cohort.csv")
    parser.add_argument('--num_cohort', type=int, default=3000)
    args = parser.parse_args()

    data = pd.read_csv(args.data_list_path)
    utt_paths = data["utt_paths"].values
    np.random.shuffle(utt_paths)
    utt_paths = utt_paths[:args.num_cohort]
    with open(args.cohort_save_path, "w") as f:
        for item in utt_paths:
            f.write(item)
            f.write("\n")
