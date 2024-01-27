import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_avg_time = 100  # Number of iterations for calculating average values

def gen_fake_data_cont_input_cont_output(N, M):
    X = pd.DataFrame(np.random.randn(N, M))
    y = pd.Series(np.random.randn(N))
    return X, y

def gen_fake_data_cont_input_cat_output(N, M):
    X = pd.DataFrame(np.random.randn(N, M))
    y = pd.Series(np.random.randint(5, size=N), dtype="category")
    return X, y

def gen_fake_data_cat_input_cat_output(N, M):
    X = pd.DataFrame({i: pd.Series(np.random.randint(2, size=N), dtype="category") for i in range(M)})
    y = pd.Series(np.random.randint(5, size=N), dtype="category")
    return X, y

def gen_fake_data_cat_input_cont_output(N, M):
    X = pd.DataFrame({i: pd.Series(np.random.randint(2, size=N), dtype="category") for i in range(M)})
    y = pd.Series(np.random.randn(N))
    return X, y

def measure_runtime_complexity(data_gen, title, subplot_index):
    learn_time = []
    predict_time = []
    vals_of_N = []
    vals_of_M = []

    for N in range(1, 6):
        vals_of_N.append(N)
        
        for M in range(1, 6):
            vals_of_M.append(M)

            X, y = data_gen(N, M)

            # Measure time for learning
            learn_temp = np.zeros(num_avg_time)
            predict_temp = np.zeros(num_avg_time)

            for i in range(num_avg_time):
                model = DecisionTree(criterion="information_gain", max_depth=5)
                
                # Measure time for learning
                start_learn_time = time.time()
                model.fit(X, y)
                end_learn_time = time.time()
                learn_temp[i] = end_learn_time - start_learn_time

                # Measure time for predicting
                start_pred_time = time.time()
                _ = model.predict(X)  # Assuming X_test is the same as X for simplicity
                end_pred_time = time.time()
                predict_temp[i] = end_pred_time - start_pred_time

            learn_time.append(np.mean(learn_temp))
            predict_time.append(np.mean(predict_temp))

    # Plot the results
    plt.subplot(2, 2, subplot_index)
    plt.plot(vals_of_M, learn_time, label="Learning Time")
    plt.plot(vals_of_M, predict_time, label="Prediction Time")
    plt.title(title)
    plt.xlabel("Number of features (M)")
    plt.ylabel("Time taken")
    plt.legend()

# Function calls
plt.figure(figsize=(12, 8))

measure_runtime_complexity(gen_fake_data_cont_input_cont_output, "Cont Input, Cont Output", 1)
measure_runtime_complexity(gen_fake_data_cont_input_cat_output, "Cont Input, Cat Output", 2)
measure_runtime_complexity(gen_fake_data_cat_input_cat_output, "Cat Input, Cat Output", 3)
measure_runtime_complexity(gen_fake_data_cat_input_cont_output, "Cat Input, Cont Output", 4)

plt.tight_layout()
plt.show()
