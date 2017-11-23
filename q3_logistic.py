from utils import load_train_small, load_train, load_valid, load_test, sigmoid
import numpy as np
from logistic_regression_template import run_logistic_regression, run_check_grad
import matplotlib.pyplot as plt
import numpy as np
from plot_digits import plot_digits

"""
CSC 2515 - Assignment 1
Tausif Sharif

Notes:
	- Currently set to show and save Cross Entropy plot for mnist_train set
	- To view small set, comment out 1st plt block and uncomment the 2nd; Rerun script
"""

hyperparameters = {
                    'learning_rate': 0.05,
                    'weight_regularization': False, # boolean, True for using Gaussian prior on weights
                    'num_iterations': 3000,
                    'weight_decay': 0 # related to standard deviation of weight prior 
                    }

#logging = run_check_grad(hyperparameters)
logging = run_logistic_regression(hyperparameters)

iterations = np.arange(1, (hyperparameters['num_iterations'] + 1))

#For mnist_train
plt.plot(iterations, logging[:, 1], label = 'Training Set (mnist_train)', color = 'r')
plt.plot(iterations, logging[:, 3], label = 'Validation Set', color = 'b', linestyle = '--')
plt.xlabel('Iterations')
plt.ylabel('Cross Entropy')
plt.title('Change in Cross Entropy on mnist_train Set')
plt.legend()
plt.savefig('q3_2_ce_mnist_train.png')
plt.show()

#For mnist_train_small (CHANGE INPUT DATA TO SMALL SET)
"""
plt.plot(iterations, logging[:, 1], label = 'Training Set (mnist_train_small)', color = 'g', linewidth = 3)
plt.plot(iterations, logging[:, 3], label = 'Validation Set', color = 'b', linestyle = '--')
plt.xlabel('Iterations')
plt.ylabel('Cross Entropy')
plt.title('Change in Cross Entropy on mnist_train_small Set')
plt.legend()
plt.savefig('q3_2_ce_mnist_train_small.png')
plt.show()
"""