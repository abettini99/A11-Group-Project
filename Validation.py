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
yyzz = []
i = 0

#Append both the node number and the x,y,z coordinate respectively to the 
#seperate lists xx,yy,zz
for i in range(len(nodes)):
    xx.append(nodes[i][:2])
    yy.append([nodes[i][0]]+[nodes[i][2]])
    zz.append([nodes[i][0]]+[nodes[i][3]])
    yyzz.append([nodes[i][0]]+[nodes[i][2]]+[nodes[i][3]])
    i = i+1

#Below the airfoil is plotted
nodes = np.array(nodes)
x,y,z = nodes[:,1], nodes[:,2], nodes[:,3]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x,y,z, c=z, cmap='Greens')
plt.show()

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

yyzz = np.array(yyzz)

# The nodes at the hinge line are generated in a list
hinge_line = []
i=0

for i in range(len(yyzz)):        
    if yyzz[i,1] == 0 and yyzz[i,2] == 0:
        hinge_line.append(i+1)
        i = i+1
    else:
        i = i+1


# here, the leading and trailing edge nodes are put into a list
le = []
i=0
for i in range(len(zz)):
    if zz[i,1] == zmax:
        le.append(i+1)
        i = i+1
    else:
        i = i+1

te = []
i=0
for i in range(len(zz)):
    if zz[i,1] == zmin:
        te.append(i+1)
        i = i+1
    else:
        i = i+1

# Create list or array of x locations corresponding to the nodes of the hingline
xloc_hinge = []

i=0
p=0

for i in range(len(hinge_line)):
    for p in range(len(xx)):
        if hinge_line[i] == xx[p,0]:
            xloc_hinge.append(xx[p,:2])
            p = p+1
        else:
            p = p+1
    i = i+1
xloc_hinge = np.array(xloc_hinge)    
xloc_hinge = xloc_hinge[xloc_hinge[:, 1].argsort()] 

#Create list or array of x locations corresponding to the nodes of the LE and
#TE
xloc_LE = []

i=0
p=0

for i in range(len(le)):
    for p in range(len(xx)):
        if le[i] == xx[p,0]:
            xloc_LE.append(xx[p,:2])
            p = p+1
        else:
            p = p+1
    i = i+1
xloc_LE = np.array(xloc_LE)    
xloc_LE = xloc_LE[xloc_LE[:, 1].argsort()] 



xloc_TE = []
i=0
p=0

for i in range(len(te)):
    for p in range(len(xx)):
        if te[i] == xx[p,0]:
            xloc_TE.append(xx[p,:2])
            p = p+1
        else:
            p = p+1
    i = i+1
xloc_TE = np.array(xloc_TE)    
xloc_TE = xloc_TE[xloc_TE[:, 1].argsort()] 



#von misses and normal stresses jam straight
stresses_jam_straight1=np.genfromtxt('B737.rpt', skip_header=13390, skip_footer=59962-19168-134)
stresses_jam_straight2=np.genfromtxt('B737.rpt', skip_header=19186, skip_footer=59962-20042-127)

stresses_jam_straight=np.vstack((stresses_jam_straight1,stresses_jam_straight2))

#deflections jam straight
deflections_jam_straight=np.genfromtxt('B737.rpt', skip_header=33374, skip_footer=59962-39962-77)
deflection_y=deflections_jam_straight[:,[0,3]]   

#bending deflections
deflections_bending=np.genfromtxt('B737.rpt', skip_header=20074, skip_footer=59962-26662-115)
deflection_y_bending=deflections_bending[:,[0,3]] 

deflection_list_hinge = []
i = 0
p = 0
for i in range(len(xloc_hinge)):
    for p in range(len(deflection_y_bending)):
        if xloc_hinge[i,0] == deflection_y_bending[p,0]:
            deflection_list_hinge.append(deflection_y_bending[p,1])
            p = p+1
        else:
            p = p+1
    i = i+1


deflection_list_LE = []
i = 0
p = 0
for i in range(len(xloc_LE)):
    for p in range(len(deflection_y_bending)):
        if xloc_LE[i,0] == deflection_y_bending[p,0]:
            deflection_list_LE.append(deflection_y_bending[p,1])
            p = p+1
        else:
            p = p+1
    i = i+1

deflection_list_TE = []
i = 0
p = 0
for i in range(len(xloc_TE)):
    for p in range(len(deflection_y_bending)):
        if xloc_TE[i,0] == deflection_y_bending[p,0]:
            deflection_list_TE.append(deflection_y_bending[p,1])
            p = p+1
        else:
            p = p+1
    i = i+1


plt.plot(xloc_TE[:,1], deflection_list_hinge)
plt.show()
    
 