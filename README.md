========================================================================================================================
Cross-Validation for Hyperparameter and Model Selection on Fashion-MNIST
========================================================================================================================
Student      : Ming Chen
Date         : July 1, 2026
========================================================================================================================


1. CONTENTS OF THIS SUBMISSION
------------------------------------------------------------------------------------------------------------------------
code.zip
  |-- README.txt                  This file.
  |
  |-- assi2_DataPrep.py           Shared data-preparation module. Loads Fashion-MNIST, restricts to classes 5 (Sandal)
  |                               and 7 (Sneaker) relabelled as 0/1, subsamples 3000 examples per class, reproduces 
  |                               features by two techniques, and flips training with noisy probability.
  |
  |-- mnist_reader.py             Fashion-MNIST loader taken from the Zalando Research repository.
  |                               (See Attribution section below.)
  |
  |-- Linear SVM.py               Part 1 experiment, trains linear SVMs and selects the best C by 5-fold CV, then
  |                               plot error results out. Appends the final test error to model_results.txt.
  |
  |-- RBF (Gaussian) SVM.py       Part 2 experiment, trains Gaussian-kernel SVMs on the unit-norm features over a 
  |                               (gamma, C) grid, and select the best pairs by 5-fold CV. Produces the error-vs-gamma 
  |                               plot and appends the final test error to model_results.txt.
  |
  |-- Neural Network.py           Part 3 experiment, trains Neural Network across 15 configurations, and selects the  
  |                               best configuration by 5-fold CV. Then produces the corresponding plots. Appends the 
  |                               final test error to model_results.txt.
  |
  |-- Tuned Model Comparison.py   Part 4 comparison experiment, reads the three test errors written by the previous 
  |                               scripts, computes 95% confidence intervals on test dataset, plots the comparison of
  |                               pairwise CI-overlap information.
  |
  |-- model_results.txt           Created at runtime. Holds the three tuned test errors (linear SVM, Gaussian SVM,
  |                               neural network) that Tuned_Model_Comparison.py consumes. 
  |
  |-- data/                       Fashion-MNIST dataset in its original gzipped IDX format. Must be present for
  |      |-- train-images-idx3-ubyte.gz    assi2_DataPrep.py / mnist_reader.py to load the data.
  |      |-- train-labels-idx1-ubyte.gz    
  |      |-- t10k-images-idx3-ubyte.gz
  |      |-- t10k-labels-idx1-ubyte.gz
report.pdf                        Submitted separately (not inside code.zip).


2. HOW TO RUN
------------------------------------------------------------------------------------------------------------------------
Requirements: Python 3.9+, numpy, pandas, scikit-learn, matplotlib.

  pip install numpy pandas scikit-learn matplotlib

Place all .py files and the data/ folder in the same directory. From that directory, run the four experiment scripts
in the order below. Each of the first three scripts appends its final tuned test error to model_results.txt, and the
fourth script reads that file to produce the comparison, so the order matters:

  python Linear_SVM.py
  python "RBF__Gaussian__SVM.py"
  python Neural_Network.py
  python Tuned_Model_Comparison.py

All scripts use a fixed seed (123) for reproducibility: the same subsampled training set, the same label-noise mask
(p = 0.2 applied only to training labels), and the same 5-fold CV splits are used across every experiment. Metrics
are printed to the console and figures are displayed via matplotlib; close each figure window to proceed. Note that
if you re-run the pipeline from scratch, delete model_results.txt first so that stale test errors from a previous run
are not carried into the Part 4 comparison.


3. ATTRIBUTION
------------------------------------------------------------------------------------------------------------------------
Dataset
  Fashion-MNIST by Zalando Research (https://github.com/zalandoresearch/fashion-mnist). The four gzipped IDX files in
  the data/ folder are the original training and test splits distributed by that repository. Only class 5 (Sandal) and
  class 7 (Sneaker) are used, relabelled as 0 and 1 respectively, as required by the assignment.

Fashion-MNIST loader (upstream code)
  mnist_reader.py is taken as-is from the Fashion-MNIST repository
  (https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py). No modifications were made.

Libraries
  Linear SVMs (LinearSVC), Gaussian-kernel SVMs (SVC), and neural networks (MLPClassifier), together with 5-fold
  cross-validation utilities (StratifiedKFold, cross_validate), are used from scikit-learn as-is; no scikit-learn
  source was modified.

Student-written code
  All remaining code (assi2_DataPrep.py, Linear_SVM.py, RBF__Gaussian__SVM.py, Neural_Network.py,
  Tuned_Model_Comparison.py, including the CV loops, the (gamma, C_gamma) tuning procedure for the Gaussian SVM,
  the vary-one-hyperparameter experiments for the neural network, the 95% Wald confidence-interval computation,
  and all plotting) was written by the student.
========================================================================================================================