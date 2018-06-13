import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x):
    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rosenbrock_grad(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der

plot_values = []

def minimize(f, f_grad, x, step=1e-3, iterations=10000, precision=1e-3):
    for i in range(0,iterations):
        plot_values.append(rosenbrock(x))
        x = x - step * f_grad(x)
    return x


x0 = np.random.rand(20)
for i,x in enumerate(x0):
    x0[i] = x * -1

print(x0)
print(rosenbrock(x0))
final_x = minimize(rosenbrock,rosenbrock_grad,x0, 1e-3)
print(final_x)
print(rosenbrock(final_x))

plt.plot(plot_values)
plt.ylabel('Rosenbrock values')
plt.show()