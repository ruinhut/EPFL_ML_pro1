# Project 1

## Generate the submission

### 1. Environment

The verified environment is Python 3.11.7, NumPy 1.26.4 and Matplotlib 3.8.0.
The learning algorithms use NumPy and the Python standard library.
Matplotlib is used for plotting; shared utilities import it when the modules are loaded.

```powershell
python -m pip install numpy==1.26.4 matplotlib==3.8.0
```

### 2. Generate the model

Place `x_train.csv`, `y_train.csv` and `x_test.csv` in `../data/dataset/`.
Run `pipeline.ipynb` from top to bottom.
The selected model is the section 5 L2 NN: 296 raw features expanded with one-hot encoding
and missing indicators, one hidden layer with 28 ReLU units, and L2 coefficient 0.0001.

Full training refits preprocessing and weights on all labeled rows using the selected
epoch count. Save model writes the weights, preprocessing rules, raw feature indices and
prediction threshold to `output/nn_model.npz`.

### 3. Generate the CSV

Run:

```powershell
python -B run.py
```

`run.py` loads `output/nn_model.npz`, reads only `x_test.csv`, and writes
`output/submit.csv` using `helpers.create_csv_submission`.
An existing saved model can be used directly without rerunning the notebook.
The CSV header is `Id,Prediction`, with one prediction in {-1, +1} for each of the
109,379 test observations in the original test-ID order.

## Selected model

The submission uses the NN that previously obtained online F1 0.454.

| Setting | Value |
| --- | --- |
| Raw features | 296 |
| Representation | One-hot encoding and missing indicators |
| Full-data inputs, including bias | 1,612 |
| Hidden layer | 28 ReLU units |
| Output | Sigmoid |
| Loss | Class-weighted binary cross-entropy plus squared L2 |
| L2 coefficient | 0.0001; biases excluded |
| Training epochs | 125 |
| Training batch size | 1,024 |
| Learning rate | 0.015 |
| Random seed | 2026 |
| Prediction threshold | 0.4525909721851349 |
| Test prediction batch size | 4,096 |

The class weights are fitted from training class counts using the inverse-square-root
rule in `NN.nn_class_weights`. Full-data training recomputes them on all labeled rows.
The architecture, epoch count and threshold were selected in the earlier experiments.
`run.py` uses the saved full-data model without changing these settings or setting thread environment variables.

## Data preparation and feature generation

### Feature processing

1. `helpers.load_csv_data` loads the three course CSV files and separates IDs from features.
   Feature data are converted to float32.
2. `make_splits` recreates the stratified 70/15/15 split with seed 2026.
   The 70% training subset is used to reproduce the same 296 selected raw columns.
3. `select_expanded_features` removes the named identifier, administrative and target-related
   columns, all-missing columns, constant columns and exact duplicates. This expanded
   selection retains source measurements that the earlier 285-column representation replaced.
4. `clean_column` applies the response-code rules written in `data_processing.py`.
   Other codes retain their values.
5. Categorical features are the explicitly listed columns, or integer-coded columns with
   at most 32 values unless designated continuous. One-hot encoding omits the most frequent
   reference category. Categories with fewer than 20 fitting observations and unseen
   categories use OTHER.
6. Continuous values are clipped at the 0.005/0.995 fitting quantiles, filled with the median,
   and standardized. Columns missing in the fitting data receive a missing indicator.
   Missing categorical entries use the reference category and their missing indicator.
7. The encoded NN input starts with a bias column. The split-fitted 296-column representation
   has 1,565 inputs; the full-data representation has 1,612 because category frequencies differ.

The 296-column selection is made once in PreProcessing and shared by sections 1, 2, 4 and 5.
Section 3 selects its own columns using the custom feature-label score.
During validation, preprocessing statistics are fitted on training rows and reused on
validation and holdout. Full-data refitting produces the saved preprocessing rules used on test.

### Cross-validation steps

1. **Linear SVM:** Use stratified five-fold cross-validation with seed 2026.
   Train on four folds and evaluate on the remaining fold, repeating for all five folds.
   Fit scaling on each training fold; apply those statistics to its validation fold and
   replace missing standardized values with zero. Report the mean and standard deviation
   of accuracy, balanced accuracy, precision, recall and F1 across the five folds.
2. **Approximate RBF SVM:** Use a stratified 80/20 split with seed 2026.
   Fit scaling on training rows and reuse it on validation. Use the same 128-dimensional
   Random Fourier Feature map for both subsets, with gamma=1/(2*d), where d is the number
   of selected raw features. Both SVM baselines use decision threshold zero.
3. **Relation-selected MSE and sigmoid NNs:** Use a stratified 80/20 split with seed 5920.
   Fit the custom score and scaling on training rows. Retain absolute scores above 0.01
   (212 columns in the current run). Keep the weights with minimum validation loss, then
   search 51 score quantiles for the threshold maximizing 0.1*accuracy + 0.9*F1.
4. **Encoded L2 logistic regression and L2 NN:** Share the stratified 70/15/15 split
   with seed 2026. Fit encoding, filling and scaling on the 70% training subset.
   At epoch 1 and every five epochs, select weights and threshold by validation F1,
   grouping tied scores in the threshold search. Evaluate holdout using that selected
   pair. Holdout and online results informed development and are not untouched estimates.

## Files

| File | Purpose |
| --- | --- |
| `implementations.py` | Implement the six required course methods: linear regression with GD, SGD and normal equations, ridge regression, logistic regression and regularized logistic regression. Each returns weights and loss. |
| `helpers.py` | Load the course CSV files and separate observation IDs, features and labels; write predictions in the required `Id,Prediction` CSV format. |
| `data_processing.py` | Create stratified splits, standardize data, clean response codes, select raw columns, and fit or apply categorical encoding and missing-value processing. |
| `data_relation.py` | Select features using the custom feature-label score; fit PCA components and project data using fitted statistics. |
| `logistic.py` | Compute L2 logistic loss and gradients, train with validation-F1 selection, predict probabilities, and refit on all labeled rows. |
| `SVM.py` | Compute hinge-loss gradients and train linear SVMs; provide RBF kernel calculations and Random Fourier Feature fitting and transformation. |
| `NN.py` | Implement forward passes, gradients and training for MSE, sigmoid and encoded L2 NNs; provide prediction and full-data refitting functions. |
| `basic_fuctions.py` | Initialize weights, calculate classification metrics, search F1 thresholds, print results and plot loss curves. |
| `pipeline.ipynb` | Load and preprocess data, train and compare the five retained method groups, refit the selected NN on all labeled rows, and save its model bundle. |
| `run.py` | Load the saved NN and preprocessing rules, predict test observations in their original order, and generate `output/submit.csv`. |
