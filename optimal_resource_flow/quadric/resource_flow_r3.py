import numpy as np #	Constraints check

def inequality_constraint(x, A, S):

    return np.dot(A, x) - S



#	Quadric problem solution

def solve_quadratic_resource_allocation(c, D, A, S, E,S_m, V, W):

    n = c.shape[0]





lambda_val = S_m / np.dot(S_m, S_m)



#	Initial values

x = np.ones(n)

S_t = S_m



while True:

#	Checking the validity of the solution

    inequality_res = inequality_constraint(x, A, S)

if np.all(inequality_res <= 0):

    return x



#	Descent directions

gamma	= np.zeros((n, n))
N_B	=	[]	
for	k	in	range(n):
    for l in range (k+1,n):
        gamma[k, l] = D[k, l] - D[k, k] - D[l, l] 
        if gamma[k, l] > 0:
            N_B.append((k, l))

            

beta = float('inf')



for t in range(len(E)):

    for (k, l) in N_B:

        a_tk = E[t, k]

        a_tl = E[t, l]

if a_tk - a_tl != 0:

    S_t_e = S_m[t]

beta = min(beta, gamma[k, l] * (S_t_e

-	S_t[t]) ** 2 / (a_tk - a_tl) ** 2)



#	Descent step

h_star = np.zeros((n, n))

for (k, l) in N_B:

    a_k = E[:, k]

    a_l = E[:, l]

if np.sum(gamma[k, l] * (a_k - a_l) ** 2) !=0:

    h_star[k, l] = beta * np.sum((a_k - a_l)*	(S_t - S_new)) / np.sum(gamma[k, l] * (a_k - a_l) ** 2) h_star[l, k] = -h_star[k, l]




S_new = S_t - np.dot(h_star, E.T)
#Solution renewal
x_new = np.zeros(n)

for i in range(n):

x_new[i] = max(0, x[i] + V[i] + W[i] * (S_new[i] - S_t[i]) / lambda_val[i])



#	Check for convergence

if np.all(x_new == x) and np.all(S_new == S_t)

and np.all(inequality_res <= 0):

return x_new



#	Solution renewal

x = x_new

S_t = S_new



#	Given parameters

c = np.array([1, 2, 3])

D	= np.array([[-2, 0, 0], [0, -3, 0], [0, 0, -1]])

A	= np.array([[1, 0, 1], [0, 1, 1]])

S = np.array([4, 3])

E	= np.array([[1, 1, 1]])

S_m = np.array([4.46])

V	= np.array([0, 0, 0])

W	= np.array([2, 3, 1])



#	Solution

result = solve_quadratic_resource_allocation(c, D, A, S, E, S_m, V, W)

print("Optimal solution:", result)




obj_value = np.dot(c, result)
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



#	Objective function

def objective_function(x, c, D):

return 0.5 * np.dot(np.dot(x.T, D), x) - np.dot(c, x)



#	

c = np.array([1, 2, 3])

D	= np.array([[-2, 0, 0], [0, -3, 0], [0, 0, -1]])



#	Range of x and y

x	= np.linspace(-5, 5, 100)

y	= np.linspace(-5, 5, 100)

X, Y = np.meshgrid(x, y)





Z	= np.zeros_like(X) for i in range(len(x)):

for j in range(len(y)):

Z[i, j] = objective_function(np.array([X[i, j], Y[i, j], X[i, j]]), c, D)



#	3D plot output

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis')


print("Objective value:", obj_value)
#	

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.set_zlabel('Objective function')

ax.set_title('Objective function')



#	Optimal point selection

result = np.array([1.67, 2.31, 0.48])

optimal_point = objective_function(result, c, D)

ax.scatter(result[0], result[1], optimal_point, color='red', label='Optimal Point')

ax.legend()





plt.show()



