import numpy as np
import matplotlib.pyplot as plt

def log_normal(mean,sigma,t):
    if t == 0:
        return 0
    return 1/(t*sigma*np.sqrt(2*np.pi))*np.exp(-((np.log(t)-mean)**2)/(2*sigma**2))


mean_normal = 1.151
sigma_normal = 0.3129

sigma = np.sqrt(np.log(1 + (sigma_normal**2)/(mean_normal**2)))
mean = np.log(mean_normal) - 0.5*sigma**2
print(mean)

t = np.linspace(0.3,2.5,100)
cost  =[]
for catalyst in t:
    cost.append(log_normal(mean,sigma,catalyst))

plt.figure()
plt.plot(t,cost)
plt.show()
