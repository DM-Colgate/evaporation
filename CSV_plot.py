#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

# set up some ploting stuff
fig = plt.figure(figsize = (9,6))
plt.style.use('fast')
pallette = plt.get_cmap('magma')
pallette1 = plt.get_cmap('viridis')


# with open('py7.dat') as file:
#     lines = file.readlines()
#     x = [float(line.split()[0]) for line in lines]
#     y = [float(line.split()[1]) for line in lines]
# plt.plot(x, y, ls = '-', linewidth = 1, label='py')



# with open('TM4_725.csv') as file:
#     lines = file.readlines()
#     x = [float(line.split(',')[0]) for line in lines]
#     y = [float(line.split(',')[1]) for line in lines]
# plt.plot(x, y, label='100 $M_\odot$', c=pallette1(2/9))

with open('TM5_400.csv') as file:
    lines = file.readlines()
    x = [float(line.split(',')[0]) for line in lines]
    y = [float(line.split(',')[1]) for line in lines]
plt.plot(x, y, label='300 $M_\odot$', c=pallette1(5/9))

with open('TM6_400.csv') as file:
    lines = file.readlines()
    x = [float(line.split(',')[0]) for line in lines]
    y = [float(line.split(',')[1]) for line in lines]
plt.plot(x, y, label='1000 $M_\odot$', c=pallette1(8/9))




# with open('star2.dat') as file:
#     lines = file.readlines()
#     x = [float(line.split()[0]) for line in lines]
#     y = [float(line.split()[1]) for line in lines]
# plt.scatter(x, y, s=1, marker=',', label='5 $M_\odot$', c=pallette1(3/10))

# with open('star3.dat') as file:
#     lines = file.readlines()
#     x = [float(line.split()[0]) for line in lines]
#     y = [float(line.split()[1]) for line in lines]
# plt.scatter(x, y, s=1, marker=',', label='15 $M_\odot$', c=pallette1(4/10))

# with open('star4.dat') as file:
#     lines = file.readlines()
#     x = [float(line.split()[0]) for line in lines]
#     y = [float(line.split()[1]) for line in lines]
# plt.scatter(x, y, s=1, marker=',', label='20 $M_\odot$', c=pallette1(5/10))

# with open('star5.dat') as file:
#     lines = file.readlines()
#     x = [float(line.split()[0]) for line in lines]
#     y = [float(line.split()[1]) for line in lines]
# plt.scatter(x, y, s=1, marker=',', label='100 $M_\odot$', c=pallette1(6/10))

# with open('star6.dat') as file:
#     lines = file.readlines()
#     x = [float(line.split()[0]) for line in lines]
#     y = [float(line.split()[1]) for line in lines]
# plt.scatter(x, y, s=1, marker=',', label='400 $M_\odot$', c=pallette1(7/10))


# with open('star7.dat') as file:
#     lines = file.readlines()
#     x = [float(line.split()[0]) for line in lines]
#     y = [float(line.split()[1]) for line in lines]
# plt.scatter(x, y, s=1, marker=',', label='1000 $M_\odot$', c=pallette1(8/10))


plt.xlabel('$m_\chi$ [GeV]')
plt.ylabel('$T_{\chi}$ [K]')
# plt.title('')
plt.legend()
plt.xscale('log')
# plt.yscale('log')
plt.xlim(10**(-5), 10**5)
# plt.ylim(5*10**7, 1.5*10**8)

# fig = plt.figure()
plt.tight_layout()
plt.savefig("Tchi.pdf")
# plt.show()
