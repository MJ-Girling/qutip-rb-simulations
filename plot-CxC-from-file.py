import numpy as np
import matplotlib.pyplot as plt

with open('RB simulations/pauli-channel-uab.txt', "r") as f:
    lines = f.readlines()
    x = [line.split()[0] for line in lines]
    y = [line.split()[1] for line in lines]
    del x[0]
    del y[0]
    x=map(float, x)
    y=map(float, y)

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.plot(np.linspace(0,1),np.linspace(0,1), color='lightgrey', label='Linear')
ax1.scatter(x,y,marker = 'o', label='Data')

ax1.set_xlabel(r'True value of $u_{AB\to AB}$',fontsize=10)
ax1.set_ylabel(r'RB extracted value of $u_{AB \to AB}$',fontsize=10)
ax1.set_title(r'Value of sub-unitarity $u_{AB \to AB}$ extracted over true value for random Pauli channels ($\mathcal{E}(\rho) = \sum_{ij} p_{ij} \ P_{i} \otimes P_j \ \rho \ P_{i} \otimes P_j$)',fontsize=15)

ax1.legend(fontsize = 15)
leg = ax1.legend()
plt.show()