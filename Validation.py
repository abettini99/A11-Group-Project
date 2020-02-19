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
    if line.startswith('*')==False and i < 6598: #create a list of floats of the nodes in x,y,z direction in order to work with them
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
xx = []
yy = []
zz = []
i = 0

#Append both the node number and the x,y,z coordinate respectively to the 
#seperate lists xx,yy,zz
for i in range(len(nodes)):
    xx.append(nodes[i][:2])
    yy.append([nodes[i][0]]+[nodes[i][2]])
    zz.append([nodes[i][0]]+[nodes[i][3]])
    i = i+1

#Below the airfoil is plotted
nodes = np.array(nodes)
x,y,z = nodes[:,1], nodes[:,2], nodes[:,3]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x,y,z, c=z, cmap='Greens')
plt.show

#Create a list of only the max and min numbers of z, including the node number
#which we need later on for the rpt file to check the corresping stresses and 
#such

#Get the dimensions of the B737 aileron first
xx = np.array(xx)
xmax = np.max(xx[:,1])
xmin = np.min(xx[:,1])

yy = np.array(yy)
ymax = np.max(yy[:,1])
ymin = np.min(yy[:,1])

zz = np.array(zz)
zmax = np.max(zz[:,1])
zmin = np.min(zz[:,1])

# here, the leading and trailing edge nodes are put into a list
le = []
i=0
for i in range(len(zz)):
    if zz[i,1] == zmax:
        le.append(i)
        i = i+1
    else:
        i = i+1

te = []
i=0
for i in range(len(zz)):
    if zz[i,1] == zmin:
        te.append(i)
        i = i+1
    else:
        i = i+1


#Create list or array of x locations corresponding to the nodes of the LE and
#TE
xloc_LE = []
i=0
p=0
for i in range(len(xx)):
    for p in range(len(le)):
        if xx[i,0] == le[p]:
            xloc_LE.append(xx[i,1])
            p = p+1
        else:
            p = p+1
        i = i+1
    else:
        i = i+1
        
# xloc_TE = []
# i=0
# for i in range(len(xx)):
#     if xx[i,0] == te[i]:
#         xloc_TE.append(xx[i,1])
#         i = i+1
#     else:
#         i = i+1




