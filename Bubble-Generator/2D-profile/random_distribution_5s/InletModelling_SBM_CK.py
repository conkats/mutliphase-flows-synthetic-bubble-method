# -*- coding: utf-8 -*-
"""
@author: constantinos katsamis
SBM inlet modelling script
"""
# This script was developed to make a new type of inlet modelling, specifically designed for tube bundle geometries
# i.e. treffle WP$ but not limited hereto. In this script, (large) bubble shapes are first defined by the user, 
# then randomly selected
# to give a certain mass flux over a unit time and finally written to an appropriate format to serve as a transient
# inlet model for CFD calculations in STARCCM+.

# import of utilities
import math
import numpy as np
import sys
import random  # package with random generator
from matplotlib import cm
import matplotlib.pyplot as plt

print("Starting inlet modelling script.\n")

# Geometry parameters
D=0.03*2
A= D*0.216
v_ratio=0.2
verbose=0
n_max=1000
z=D

casePath = '.' #str(sys.argv[1])
U=float(0.463)#0.589#Velocity of the mixture  to be introduced in domain 
startTime = float(0)#Time step from where inlet should be modelled
endTime = float(5)#30Last flow time to be defined
timeStepSize = float(0.01)#Time step size to be used in following calculation
tunit =0.1 #round(timeStepSize*U,2)#, should be x10 >timeStepSize #Unit time scale - parameter for inlet modelling ???
inletName = str('Inlet')#Name of the inlet boundary to be modelled
rhog = float(1.18415)##Density of the gas
rhol = float(997.561) #Density of the liquid
intersectBoundary = str(True)#Boolean indicating whether prescribed bubble shapes may intersect with the domain boundaries
intersectBubble = str(True)#Boolean indicating whether prescribed bubble shapes may intersect with previously defined bubbles

mg=0.011*tunit#rhog*A*U*v_ratio#Amount of the gas to be introduced in domain over a time 'tunit'
mg_tunit = float(A*v_ratio*rhog*U)*tunit
#tol_mg = float(mg_tunit*0.2)#Tolerenace on the amount of gas to be introduced in domain over a time 'tunit' 
tol_mg=float(1e-5)
#mgb_min = float(0.01*mg)
#mgb_max = float(0.5*mg)
#if verbose: print('min/max',mgb_min,mgb_max)


if int((endTime-startTime)/tunit) != ((endTime-startTime)/tunit):
    sys.exit("The desired time interval (endTime - startTime) should be a multiple of tunit.")
if endTime <= startTime:
    sys.exit("The endTime should be larger than the startTime.")
if (abs(int(tunit/timeStepSize) - tunit/timeStepSize) >= timeStepSize) and (abs((int(tunit/timeStepSize)+1) - tunit/timeStepSize) >= timeStepSize):
    sys.exit("Variable tunit should be a multiple of timeStepSize.")  
    
    
# Inlet geometry and normal to the inlet condition prepared with 'TubeBundle_readInlet_<CFD-programme>.py' 
#==================================================
# Creating input coordinates list. Face areas should be just 
# Need to check and match the geo of the actual inlet
#m3-220*20
nx=220
nz=20
#ny=360
x=np.linspace(-0.108,0.108,nx)
#y=np.linspace(-4.0800e-01, 4.0800e-01,ny)
z=np.linspace(0,0.06,nz)
dx=x[1]-x[0]
dz=z[1]-z[0]
xv, zv = np.meshgrid(x,z)

#construct the 2-D coordinate list
#flatten the 2D arrays into 1D arrays
xvc = xv.flatten()
zvc = zv.flatten()
yvc = np.ones(len(xvc))*(-4.0800e-01)#matching the y-coord of the inlet plane

print(len(yvc), len(xvc), len(zvc))
print(len(np.ones(len(zvc))*dx*dz),len(np.arange(len(xvc))))
print(xvc.shape,yvc.shape,zvc.shape)

#idexing from 0 to len(xvc), xvc, yvc, zvc, array of face areas 1*dx*dz
coordList=np.array([np.arange(len(xvc)),xvc,yvc,zvc,np.ones(len(xvc))*dx*dz]).T
normalInlet=np.array([0,1,0])#*dxy*dxz
#==================================================

# Initializing U and VOFw
nTimeSteps = int((endTime-startTime)/timeStepSize)+1 # Plus 1 because first time step is included - last time step is not included.
UVal = np.ones([len(coordList), nTimeSteps, 3])  # initially: "pre-inlet domain" at constant velocity
for i in np.arange(3):
    UVal[:, :, 0] = U*normalInlet[0]
    UVal[:, :, 1] = U*normalInlet[1]
    UVal[:, :, 2] = U*normalInlet[2]
VOFwVal = np.ones([len(coordList), nTimeSteps,1])#int(z/dxy)+1,1])  # initially: "pre-inlet domain" filled with water
timeVal = np.arange(startTime, endTime, timeStepSize)  # list of flow times to be defined in this model

# Creating bubble shapes - under the hood, so hard-coded shapes
# Bubble shapes are defined as 1 function named "bubbleShape", comprising a switch based on the shapeID of the bubble
# (each shape gets its own shapeID and is defined in another part of the switch)
# Input: centerpoint location cellID in coordList - time instant index in vector timeVal at which cell centre appears -
# integer 'timeInterval' denoting which time interval is being defined - shapeID-variable: what bubble shape you want to
# define - amount of mass of gas desired in bubble (scaling factor) - amount of gas in [0,tunit[ which still needs to be
# defined to get to mg_tunit
# Output: boolean indicating whether the randomly chosen centerpoint and bubble shape were compatible, if False no
# bubble will be defined.
Nshapes = 1  # hard-coded counter used to verify validity of input of "shapeID"-variable - adapt when adding or removing bubble shapes
probabilityShapes = [1.0] # hard-coded probability distribution of the bubble shapes - el. 0 = probability of bubble shape 0 .... // It is checked that the sum of elements equals zero.
if np.sum(probabilityShapes) != 1.0:
    sys.exit('Vector "probabilityShapes" indicating the probability of occurrence of bubble shapes has not been defined correctly.')

j_glob=0
t_glob=0
#iter=0

def bubbleShape(C_ID, C_t, timeInterval, shapeID, mgb, mg_StillRequired):
#def bubbleShape(C_ID, C_t, timeInterval, shapeID, mgb, mgb_min, mgb_max, mg_StillRequired):
    global UVal, VOFwVal # define UVal and VOFwVal to be global such that these matrices can be altered directly by this function - as coordList will not be adapted in this function, it does not need to be defined as global (Python automatically looks for coordList definition outside of function)
    
    global t_glob,j_glob
    UVal_temp = np.array(UVal)
    VOFwVal_temp = np.array(VOFwVal)
    
    UVal_temp_0=np.array(UVal_temp)
    VOFwVal_temp_0=np.array(VOFwVal_temp)
    
    C_coord = coordList[C_ID, :]
    C_time = timeVal[C_t]
    if shapeID > (Nshapes-1):
        sys.exit("Fatal error. Shape generator asks for non-existing bubble shape (Nshapes is too low or too few shapes have been defined).")
       
    # Start of switch: every switch input is another bubble shape
    # For every bubble, you can define different requirements for the cell center location and for the scaling factor,
    # which - if not met - causes the end of the current function call. However, always make sure that you have at least
    # one bubble shape that can define a bubble sufficiently small to fall within the set tolerance tol_mg (see further)
    # of the desired amount of gas mg_tunit.
    if shapeID == 0:
        rg = ((3.0*mgb)/(4.0*math.pi*rhog))**(1.0/3.0)
        print(rg,'^rg\n')
        # Check that the scaling factor is within acceptable range:
        # This spherical bubble should be able to yield small bubbles required to make sure you can come within tol_mg
        # of the desired mg_tunit without overshooting it. That is why, if the required amount of gas is lower than the
        # normal minimum for the gas bubble, mgb is just set to mgb_StillRequired
        mgb_min=0.05*mg_tunit 
        mgb_max=0.2*mg_tunit
        if mgb_min > mg_StillRequired:
            mgb_min = 0.0
            mgb = mg_StillRequired
        if mgb < mgb_min:
            return False, 0.0
        elif mgb > mgb_max:
            return False, 0.0
        # If desired, check that the center point denoted by C_ID and C_t follows a certain set of requirements
        C_checked = True
        timeLoc = C_time-startTime-int((C_time-startTime)/tunit)*tunit  # how many seconds compared to start t_unit
        # Checks below prevents intersection with beginning of t_unit domain
        if timeLoc < rg/U:
            C_checked = False
        if timeLoc > (tunit-rg/U):
            C_checked = False
        if not(C_checked):  # Position of the bubble center (C_ID,C_t) is not OK.
            return False, 0.0
        # Check for each element in the VOFwVal[:, :, 0] whether it's in the bubble to be defined and whether this
        # bubble does not intersect with a previously defined gas bubble. If no old bubble is intersected, change the
        # element VOFwVal and UVal to the appropriate-value; this will be stored in the temporary matrices which will be
        # checked afterwards before updating VOFwVal and UVal
        # Concurrently, integrate the mass of gas you have introduced in the domain.
        mg_checked = 0.0
        mg_bubbleWall = 0.0
        coordCenter = np.array([C_coord[1] - (U * C_time) * normalInlet[0], C_coord[2] - (U * C_time) * normalInlet[1],
                                C_coord[3] - (U * C_time) * normalInlet[2]])

        i_mask = np.linalg.norm(coordList[:, 1:4] - C_coord[1:4], axis=1) < rg
        i_list = i_mask.nonzero()[0]

        j_min = int((timeLoc - rg/U)/timeStepSize)
        j_max = int(math.ceil((timeLoc + rg/U)/timeStepSize)) + 1
        
        # nested loop could maybe be eliminated by vectorization: boolean arithmetic on the matrices
        for i in i_list:
            for j in range(j_min, j_max):
                j_glob=j
                t_glob=t
                if t * int(tunit / timeStepSize) + j>=len(timeVal):
                    continue
                coordPoint = np.array(
                    [coordList[i, 1] - (U * timeVal[t * int(tunit / timeStepSize) + j]) * normalInlet[0],
                     coordList[i, 2] - (U * timeVal[t * int(tunit / timeStepSize) + j]) * normalInlet[1],
                     coordList[i, 3] - (U * timeVal[t * int(tunit / timeStepSize) + j]) * normalInlet[2]])
                if np.linalg.norm(coordPoint-coordCenter) < rg:
                    if VOFwVal[i, timeInterval * int(
                            tunit / timeStepSize) + j, 0] == 1.0:  # Every cell not yet occupied by bubble
                        VOFwVal_temp[i, timeInterval * int(tunit / timeStepSize) + j, 0] = 0.0
                        UVal_temp[i, timeInterval * int(tunit / timeStepSize) + j, 0] = U * normalInlet[0]
                        UVal_temp[i, timeInterval * int(tunit / timeStepSize) + j, 1] = U * normalInlet[1]
                        UVal_temp[i, timeInterval * int(tunit / timeStepSize) + j, 2] = U * normalInlet[2]
                        mg_checked = mg_checked + coordList[i, 4] * U * timeStepSize * rhog
                        mg_bubbleWall = mg_bubbleWall + coordList[i, 4] * U * timeStepSize * rhog
                    elif intersectBubble:
                        mg_bubbleWall = mg_bubbleWall + coordList[
                            i, 4] * U * timeStepSize * rhog  # In this case, a cell was already filled with air, but I will add the mass of air to mg_bubbleWall to be able to check later whether a wall was intersected.
                    else:
                        return False, 0.0
        
        #check for extra zone around the bubble. lets say 20% of radius?
        buffer_ratio=0.15 #ratio of buffer region to x.
        rg=(1+buffer_ratio)*rg
        i_mask = np.linalg.norm(coordList[:, 1:4] - C_coord[1:4], axis=1) < rg
        i_list = i_mask.nonzero()[0]
        
        j_min = int((timeLoc - rg/U)/timeStepSize)
        j_max = int(math.ceil((timeLoc + rg/U)/timeStepSize)) + 1
        
        for i in i_list:
            for j in range(j_min, j_max):
                j_glob=j
                t_glob=t
                if t * int(tunit / timeStepSize) + j>=len(timeVal):
                    continue
                coordPoint = np.array(
                    [coordList[i, 1] - (U * timeVal[t * int(tunit / timeStepSize) + j]) * normalInlet[0],
                     coordList[i, 2] - (U * timeVal[t * int(tunit / timeStepSize) + j]) * normalInlet[1],
                     coordList[i, 3] - (U * timeVal[t * int(tunit / timeStepSize) + j]) * normalInlet[2]])
                if np.linalg.norm(coordPoint-coordCenter) < rg:
                    if VOFwVal[i, timeInterval * int(
                            tunit / timeStepSize) + j, 0] == 0.0:  # If any cell occupied already by bubble
                        print('too close to other bubble')
                        return False, 0.0
        
    # Check mass of gas added to the domain: in case intersection with boundary is not allowed (intersectBoundary=False)
    print('intersect boudnary',intersectBoundary)
    if not(intersectBoundary):
        if mg_bubbleWall < (mgb-np.average(coordList[:, 4])*U*timeStepSize):
            print('too close to other bubble')
            return False, 0.0

    # Save temporary files to permanent files
    UVal = UVal_temp
    VOFwVal = VOFwVal_temp
    
    return True, mg_checked

           
           
# Selecting bubble shapes based on user input
# The random generator selects randomly: the bubble shape definition (shapeID) - the center point of a bubble, both in
# inlet plane and in time (normal direction) - the amount of mass that bubble should have
# Distribution of the center points is random through entire inlet - restriction to center point location or mass of gas
# in bubble are defined in the bubble shapes.

nIntervals = int((endTime-startTime)/tunit)  # Number of intervals [0,tunit[
print("Between startTime " + str(startTime) + "s and endTime " + str(endTime) + "s, " + str(
    nIntervals) + " intervals of " + str(tunit) + "s need to be defined.")
for t in np.arange(nIntervals):
    iter = 0
    mg_defined = 0.0 # Variable checking the amount of gas already defined
    while (mg_tunit-mg_defined) > (tol_mg):
        print(mg_tunit,'mg_defined',mg_defined)
        shapeID = random.randint(0, Nshapes - 1)  # randomly select bubble shape
        C_ID = random.randint(0, len(coordList) - 1)  # randomly select centerpoint location - determined by cell center ID (2D determined)
        C_t = random.randint(t * int(tunit / timeStepSize), (t + 1) * int(tunit / timeStepSize) - 1)  # randomly select centerpoint time location - determined by time step index in timeVal (1D determined)
        mg_bubble=(random.random())*(mg_tunit-mg_defined) # randomly select a scale factor for the bubble you are creating
        print(mg_bubble)
        #         print("Still needed: "+str(mg_tunit-mg_defined))
#         print("Proposed: "+str(mg_bubble))
        print('CID',C_ID)
        bubbleDefined,mg_checked=bubbleShape(C_ID,C_t,t,shapeID,mg_bubble,mg_tunit-mg_defined)
        #bubbleDefined, mg_checked = bubbleShape(C_ID, C_t, t, shapeID, mg_bubble, mgb_min, mgb_max, mg_tunit-mg_defined)
        print('mg_checked',mg_checked)
        print('bubbleDefined',bubbleDefined)
        if bubbleDefined:
            mg_defined = mg_defined+mg_checked
            iter = 0
        else:
            iter = iter+1
        if iter > n_max:
            sys.exit("Forced exit: " +str(n_max)+" trials of bubble definition have failed; system is ill-defined.")
    print("Time interval " + str(t) + " has been defined: "+str(mg_defined)+"kg of gas was inserted. (desired: "+str(mg_tunit)+"kg).")
print("Inlet was modelled successfully. \n")     


fig, ax = plt.subplots()
X,Y,Z=coordList[:,1],coordList[:,3],VOFwVal[:,-8,0]
ax.tricontour(X,Y,Z,levels=10,colors='k')
cntr=ax.tricontourf(X,Y,Z,cmap="RdBu_r")
fig.colorbar(cntr,ax=ax)
ax.plot(X,Y,'ko',ms=3)
ax.set(xlim=(-0.108,0.108),ylim=(0,D))
ax.set_title(f'Bubble Inlet at timestep {endTime}')

print("Saving inlet profile to CSV-files. ")
file = 'bubbles_time_001_vf_ck=0.2.csv'

#CKMOD: Approach write to file for starcmm+
#X , Y , Z , Value1 [t=0.1] , Value2 [t=0.1] , Value1 [t=0.2] , Value2 [t=0.2] ,
# Initialize storage for rows
rows = []

# Time loop
for j in range(len(timeVal)):
    #  Adjusted Coordinates for time
    coords = np.array([
        coordList[:, 1] - (U * timeVal[j]) * normalInlet[0],
        coordList[:, 2] - (U * timeVal[j]) * normalInlet[1],
        coordList[:, 3] - (U * timeVal[j]) * normalInlet[2]
    ]).T  # Shape: (nPoints, 3)

    # Compute velocity magnitude
    #vel_mag = np.sqrt(UVal[:, j, 0]**2 + UVal[:, j, 1]**2 + UVal[:, j, 2]**2)
    # Velocity magnitude
    vel_mag = np.linalg.norm(UVal[:, j, :], axis=1)

    if j == 0:
        # First timestep: Initialize with coordinates
        rows = coords.tolist()
        # Add first values: VOF and velocity magnitude
        for i in range(len(rows)):
            rows[i].append(VOFwVal[i, j, 0])
            rows[i].append(vel_mag[i])
    else:
        # Append VOF and velocity magnitude for each row
        for i in range(len(rows)):
            rows[i].append(VOFwVal[i, j, 0])
            rows[i].append(vel_mag[i])

# Write the file to CSV
with open(file, 'w') as f:
    # Header for coordlist
    header = ['X', 'Y', 'Z']
    #add time header for starccm table(xyz,time)
    for j in range(len(timeVal)):
        header.append('Water-Fraction[t='+str(timeVal[j])+'s]')
        header.append('Velocity-Magnitude[t='+str(timeVal[j])+'s]')
    f.write(','.join(header) + '\n')

    # Data rows
    for row in rows:
        f.write(','.join(str(val) for val in row) + '\n')

print("Inlet profile saved to single CSV file.")