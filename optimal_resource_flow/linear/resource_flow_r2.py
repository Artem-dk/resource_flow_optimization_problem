import numpy as np



#	Input data

c = np.array([5.3, 5.1, 5, 4.8]) 
A = np.array([[3, 3.1, 3.2, 2.8],
[2.1, 1.9, 2.2, 1.9], [1, 1, 1, 1]])

s = np.array([30.1, 20, 10])

V	= np.array([1, 1, 1, 1])

W	= np.array([5, 4, 5, 5])

t0 = 9

lambda_val = 0.2

m_tau = 20

sigma_tau = 12



#	Number of variables

n	= len(c)



#	Initializing variables

x = np.zeros(n)

t = np.zeros(n)

tau = np.zeros(n)

z = np.zeros(n)

s_t = np.zeros(n)

s_m = np.zeros(n)



#	Density function 

def phi(tau_val):
    return lambda_val * (1 - np.exp(-lambda_val *tau_val))
#	Розрахунок моментів tau

t[0] = t0

for j in range(1, n):

    tau[j]=phi(np.random.exponential(scale=1/lambda_val))



for j in range(1, n):

    t[j] = t[j-1] + tau[j]



#	Calculation of values s(t) and s(m)

for j in range(n):

    z[j] = np.random.normal(loc=m_tau, scale=sigma_tau)

s_t[j] = 1 / phi(z[j])

s_m[j] = 1 / phi(t[j])



#	Розширена задача

eps = 1e-6

A_extended = A.copy()

A_extended[0] += eps * A[1] * x[1]

A_extended[1] += eps * A[0] * x[0]



#	Extended problem solution

optimal_solution = np.linalg.lstsq(A_extended, s, rcond=None)[0]



#	Optimality check

if np.all(np.matmul(A, optimal_solution) <= s):

    solution = optimal_solution

else:
    #	Descent pitch

    indices = np.where(np.matmul(A, optimal_solution) > s)[0]

    t_indices = np.where(s_t > s_m)[0]

    alpha_t = np.max([np.min([(c[j] - np.dot(A[j], optimal_solution)) / np.dot(A[j], (s_t - s_m)[t_indices]) for j in indices])])





h	= np.minimum(np.minimum(x - V, W - x), alpha_t * (s_t - s_m))



#	Solution renewal

solution = np.where(indices[:, None] == np.arange(n), optimal_solution - h, optimal_solution)



#	Rounding off the solution

solution = np.round(solution, 4)

objective_value = np.round(np.dot(c, solution), 2)



#	Result output

print("Optimal solution:")

print(solution)

print("The value of the objective function:")

print(objective_value)
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

x_optimal = np.array([2.1849, 3.0411, 1.8767, 2.8973])
#	Objective function

def F(x):

  return 5.3*x[0] + 5.1*x[1] + 5*x[2] + 4.8*x[3]

x1 = np.linspace(1, 5, 100)

x2 = np.linspace(1, 4, 100)



X1, X2 = np.meshgrid(x1, x2)

Z = F([X1, X2, x_optimal[2], x_optimal[3]])



#	3D Plot

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')



ax.plot_surface(X1, X2, Z, cmap='viridis')

ax.set_xlabel('x[1]')

ax.set_ylabel('x[2]')

ax.set_zlabel('F(x)')

ax.set_title('3D Plot of F(x)')



#	Selecting the optimal point

ax.scatter(x_optimal[0], x_optimal[1], F(x_optimal), color='r', label='Optimal Solution')



plt.legend()

plt.show()



