# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:05:19 2020

@author: Xander
"""

import numpy as np
from math import *
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


i = 0
nodes = []
elements = []
assembly = []

inp = open('B737.inp')
for line in inp:
    if line.startswith('*')==False and i < 6598: #create a list from floats of the nodes in x,y,z direction in order to work with them
        node_list = [float(str(x)) for x in line.split(',')]       
        i = i+1
        nodes.append(node_list)
    
    elif line.startswith('*')==False and i > 6597 and i < 13233: #Create a list from floats of the elements in x,y,z direction
        element_list = [float(str(x)) for x in line.split(',')]
        i = i+1
        elements.append(element_list)
        
    elif line.startswith('*') == False and i > 14141 and i < 14178:
        assembly_list = [float(str(x)) for x in line.split(',')]
        i = i+1
        assembly.append(assembly_list)
        
    else: 
        i = i+1
        
#Now you want to know which nodes correspond to TE and LE for analysis
x = []
y = []
z = []
i = 0

#Apparently, you need to switch the y and z locations of inp file
for i in range(len(nodes)):
    x.append(nodes[i][:2])
    y.append(nodes[i][0]+nodes[i][3])
    z.append(nodes[i][0]+nodes[i][2])
    i = i+1

#Below the airfoil is plotted
nodes = np.array(nodes)
x,y,z = nodes[:,1], nodes[:,3], nodes[:,2]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x,y,z, c=z, cmap='Greens')
plt.show

# zmax = []
# for TE in nodes:
    



