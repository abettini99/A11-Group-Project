import math


C_a = 0.515
l_a = 2.691


#Extracting the Aero data from the file and putting it into a 2 dimensional list called Aerolst. 
#This list has 81 rows and 41 columns and is based in the coordinate system defined in the report.
Aero = open("aerodynamicloaddo228.dat","r")
Rawaerodata = Aero.read()
Aero.close()
Aerolines = Rawaerodata.split("\n")
Aerolst = []
for i in range(81):
    Aerolst.append([])
for i in range(81):
    line = Aerolines[i].split(",")
    for j in range(41):    
        Aerolst[80-i].append(line[j])
#-------------------------------------------
        
#This next part is about giving each aero data point an x' and z' coordinate. 
#These coordinates are given in Coordlst , which is also based in the same coordinate system. each point has a value of x and z
def z_coord(integer):
    theta_zi = integer/81*math.pi 
    theta_ziplus = (integer+1)/81*math.pi
    z = C_a - 0.5*(C_a*0.5*(1-math.cos(theta_zi))+C_a*0.5*(1-math.cos(theta_ziplus)))
    return z

def x_coord(integer):
    theta_zi = integer/41*math.pi 
    theta_ziplus = (integer+1)/41*math.pi
    x = 0.5*(l_a*0.5*(1-math.cos(theta_zi))+l_a*0.5*(1-math.cos(theta_ziplus)))
    return x

Coordlst=[]
for i in range(81):
    Coordlst.append([])
    z = z_coord(i)
    for j in range(41):
        x = x_coord(j)
        coord = x,z
        Coordlst[i].append(coord)

#---------------------------------------------


    
#def x_coord(Integer):
    
    
