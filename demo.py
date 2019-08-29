from dhsic import dhsic_test, dhsic
import numpy as np

def demo():
	print('case of 3 jointly independent variables.')
	X = np.random.randn(100, 3)
	print(dhsic_test(X), '\n')

	print('case of variable 3 depending on variable 1 and 2.')
	X[:,2] = X[:,0] + X[:,1]
	print(dhsic_test(X))

if __name__ == '__main__':
	demo()