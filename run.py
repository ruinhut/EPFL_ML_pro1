import os
import numpy as np

from helpers import create_csv_submission
from data_processing import transform_nn_input
from NN import predict_encoded_nn


def main():
    """Load the saved NN and generate output/submit.csv."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    test_file = os.path.join(script_dir, "..", "data", "dataset", "x_test.csv")

    with np.load(os.path.join(output_dir, "nn_model.npz"), allow_pickle=True) as model:
        w1 = model["w1"]
        w2 = model["w2"]
        raw_feature_indices = model["raw_feature_indices"]
        schema = model["schema"].item()
        threshold = model["threshold"].item()

    test_data = np.genfromtxt(test_file, delimiter=",", skip_header=1)
    test_ids = test_data[:, 0].astype(int)
    x_test = test_data[:, 1:].astype(np.float32)
    predictions = np.empty(x_test.shape[0], dtype=int)

    batch_size = 4096
    for start in range(0, x_test.shape[0], batch_size):
        end = min(start + batch_size, x_test.shape[0])
        test_raw = x_test[start:end][:, raw_feature_indices]
        test_tx = transform_nn_input(test_raw, schema, "onehot_missing")
        scores = predict_encoded_nn(test_tx, w1, w2)
        predictions[start:end] = np.where(scores >= threshold, 1, -1)

    output_file = os.path.join(output_dir, "submit.csv")
    create_csv_submission(test_ids, predictions, output_file)
    print("Submission saved:", output_file)
    print("Test prediction counts:", np.unique(predictions, return_counts=True))
    return predictions


if __name__ == "__main__":
    main()
