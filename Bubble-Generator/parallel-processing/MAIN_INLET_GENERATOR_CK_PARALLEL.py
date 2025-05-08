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
# import of utilities
import math
import numpy as np
import sys
import random  # package with random generator
from matplotlib import cm
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
import time

def print_info(message):
    """Print with process info for debugging"""
    print(f"Process {mp.current_process().name}: {message}")

print("Starting inlet modelling script with multiprocessing.\n")

# Configuration parameters
D = 0.03*2
A = D*0.216
v_ratio = 0.2
verbose = 0
n_max = 1000
z = D

casePath = '.'
U = float(0.463)
startTime = float(0)
endTime = float(1)
timeStepSize = float(0.01)
tunit = 0.1
inletName = str('Inlet')
rhog = float(1.18415)
rhol = float(997.561)
intersectBoundary = str(False)
intersectBubble = str(False)

mg = 0.011*tunit
mg_tunit = float(A*v_ratio*rhog*U)*tunit
tol_mg = float(1e-5)

if int((endTime-startTime)/tunit) != ((endTime-startTime)/tunit):
    sys.exit("The desired time interval (endTime - startTime) should be a multiple of tunit.")
if endTime <= startTime:
    sys.exit("The endTime should be larger than the startTime.")
if (abs(int(tunit/timeStepSize) - tunit/timeStepSize) >= timeStepSize) and (abs((int(tunit/timeStepSize)+1) - tunit/timeStepSize) >= timeStepSize):
    sys.exit("Variable tunit should be a multiple of timeStepSize.")  

# Inlet geometry and normal to the inlet condition
nx = 220
nz = 20
x = np.linspace(-0.108, 0.108, nx)
z = np.linspace(0, 0.06, nz)
dx = x[1]-x[0]
dz = z[1]-z[0]
xv, zv = np.meshgrid(x, z)

# Construct the 2-D coordinate list
xvc = xv.flatten()
zvc = zv.flatten()
yvc = np.ones(len(xvc))*(-4.0800e-01)  # matching the y-coord of the inlet plane

print(f"Coordinates: {len(yvc)}, {len(xvc)}, {len(zvc)}")
print(f"Data sizes: {len(np.ones(len(zvc))*dx*dz)}, {len(np.arange(len(xvc)))}")
print(f"Shapes: {xvc.shape}, {yvc.shape}, {zvc.shape}")

# Indexing from 0 to len(xvc), xvc, yvc, zvc, array of face areas 1*dx*dz
coordList = np.array([np.arange(len(xvc)), xvc, yvc, zvc, np.ones(len(xvc))*dx*dz]).T
normalInlet = np.array([0, 1, 0])

# Initializing U and VOFw
nTimeSteps = int((endTime-startTime)/timeStepSize) + 1  # Plus 1 because first time step is included
timeVal = np.arange(startTime, endTime, timeStepSize)  # list of flow times to be defined in this model

# Creating bubble shapes
Nshapes = 1  # hard-coded counter used to verify validity of input of "shapeID"-variable
probabilityShapes = [1.0]  # hard-coded probability distribution of the bubble shapes
if np.sum(probabilityShapes) != 1.0:
    sys.exit('Vector "probabilityShapes" indicating the probability of occurrence of bubble shapes has not been defined correctly.')

def bubbleShape(UVal, VOFwVal, coordList, timeVal, normalInlet, timeStepSize, rhog, U, tunit, t, C_ID, C_t, timeInterval, shapeID, mgb, mg_StillRequired):
    """
    Define bubble shape at a given point and time
    Now accepts UVal and VOFwVal as parameters instead of using globals
    """
    UVal_temp = np.array(UVal)
    VOFwVal_temp = np.array(VOFwVal)
    
    C_coord = coordList[C_ID, :]
    C_time = timeVal[C_t]
    if shapeID > (Nshapes-1):
        print_info(f"Fatal error. Shape generator asks for non-existing bubble shape (Nshapes is too low or too few shapes have been defined).")
        return False, 0.0, UVal, VOFwVal
       
    if shapeID == 0:
        rg = ((3.0*mgb)/(4.0*math.pi*rhog))**(1.0/3.0)
        print_info(f"rg: {rg}")
        
        # Check that the scaling factor is within acceptable range
        mgb_min = 0.05*mg_tunit 
        mgb_max = 0.2*mg_tunit
        if mgb_min > mg_StillRequired:
            mgb_min = 0.0
            mgb = mg_StillRequired
        if mgb < mgb_min:
            return False, 0.0, UVal, VOFwVal
        elif mgb > mgb_max:
            return False, 0.0, UVal, VOFwVal
            
        # Check that the center point requirements
        C_checked = True
        timeLoc = C_time-startTime-int((C_time-startTime)/tunit)*tunit  # how many seconds compared to start t_unit
        
        # Checks to prevent intersection with beginning of t_unit domain
        if timeLoc < rg/U:
            C_checked = False
        if timeLoc > (tunit-rg/U):
            C_checked = False
        if not C_checked:  # Position of the bubble center is not OK
            return False, 0.0, UVal, VOFwVal
            
        # Check for each element in VOFwVal whether it's in the bubble to be defined
        mg_checked = 0.0
        mg_bubbleWall = 0.0
        coordCenter = np.array([C_coord[1] - (U * C_time) * normalInlet[0], 
                              C_coord[2] - (U * C_time) * normalInlet[1],
                              C_coord[3] - (U * C_time) * normalInlet[2]])

        i_mask = np.linalg.norm(coordList[:, 1:4] - C_coord[1:4], axis=1) < rg
        i_list = i_mask.nonzero()[0]

        j_min = int((timeLoc - rg/U)/timeStepSize)
        j_max = int(math.ceil((timeLoc + rg/U)/timeStepSize)) + 1
        
        # Process cells within bubble
        for i in i_list:
            for j in range(j_min, j_max):
                if t * int(tunit / timeStepSize) + j >= len(timeVal):
                    continue
                    
                coordPoint = np.array(
                    [coordList[i, 1] - (U * timeVal[t * int(tunit / timeStepSize) + j]) * normalInlet[0],
                     coordList[i, 2] - (U * timeVal[t * int(tunit / timeStepSize) + j]) * normalInlet[1],
                     coordList[i, 3] - (U * timeVal[t * int(tunit / timeStepSize) + j]) * normalInlet[2]])
                     
                if np.linalg.norm(coordPoint-coordCenter) < rg:
                    if VOFwVal[i, timeInterval * int(tunit / timeStepSize) + j, 0] == 1.0:  # Cell not yet occupied by bubble
                        VOFwVal_temp[i, timeInterval * int(tunit / timeStepSize) + j, 0] = 0.0
                        UVal_temp[i, timeInterval * int(tunit / timeStepSize) + j, 0] = U * normalInlet[0]
                        UVal_temp[i, timeInterval * int(tunit / timeStepSize) + j, 1] = U * normalInlet[1]
                        UVal_temp[i, timeInterval * int(tunit / timeStepSize) + j, 2] = U * normalInlet[2]
                        mg_checked = mg_checked + coordList[i, 4] * U * timeStepSize * rhog
                        mg_bubbleWall = mg_bubbleWall + coordList[i, 4] * U * timeStepSize * rhog
                    elif intersectBubble == "True":
                        mg_bubbleWall = mg_bubbleWall + coordList[i, 4] * U * timeStepSize * rhog
                    else:
                        return False, 0.0, UVal, VOFwVal
        
        # Check for extra zone around the bubble (buffer region)
        buffer_ratio = 0.15  # ratio of buffer region to x
        rg_buffer = (1+buffer_ratio)*rg
        i_mask = np.linalg.norm(coordList[:, 1:4] - C_coord[1:4], axis=1) < rg_buffer
        i_list = i_mask.nonzero()[0]
        
        j_min = int((timeLoc - rg_buffer/U)/timeStepSize)
        j_max = int(math.ceil((timeLoc + rg_buffer/U)/timeStepSize)) + 1
        
        for i in i_list:
            for j in range(j_min, j_max):
                if t * int(tunit / timeStepSize) + j >= len(timeVal):
                    continue
                    
                coordPoint = np.array(
                    [coordList[i, 1] - (U * timeVal[t * int(tunit / timeStepSize) + j]) * normalInlet[0],
                     coordList[i, 2] - (U * timeVal[t * int(tunit / timeStepSize) + j]) * normalInlet[1],
                     coordList[i, 3] - (U * timeVal[t * int(tunit / timeStepSize) + j]) * normalInlet[2]])
                     
                if np.linalg.norm(coordPoint-coordCenter) < rg_buffer:
                    if VOFwVal[i, timeInterval * int(tunit / timeStepSize) + j, 0] == 0.0:  # If cell occupied already by bubble
                        print_info('Too close to other bubble')
                        return False, 0.0, UVal, VOFwVal
    
    # Check mass of gas added to the domain: in case intersection with boundary is not allowed
    print_info(f'Intersect boundary: {intersectBoundary}')
    if intersectBoundary == "False":
        if mg_bubbleWall < (mgb-np.average(coordList[:, 4])*U*timeStepSize):
            print_info('Too close to boundary')
            return False, 0.0, UVal, VOFwVal

    # Return updated matrices
    return True, mg_checked, UVal_temp, VOFwVal_temp

def try_place_bubble(args):
    """Worker function for parallel processing"""
    (t, mg_needed, shared_data) = args
    UVal, VOFwVal, coordList, timeVal, normalInlet = shared_data
    
    # Try to place bubbles for this time interval until we've added enough gas or reached max iterations
    local_mg_defined = 0.0
    local_UVal = np.array(UVal)
    local_VOFwVal = np.array(VOFwVal)
    local_iter = 0
    successful_bubbles = []
    
    print_info(f"Starting time interval {t}, need to place {mg_needed} kg of gas")
    
    while (mg_needed - local_mg_defined) > tol_mg and local_iter < n_max:
        shapeID = random.randint(0, Nshapes - 1)  # randomly select bubble shape
        C_ID = random.randint(0, len(coordList) - 1)  # randomly select centerpoint location
        C_t = random.randint(t * int(tunit / timeStepSize), (t + 1) * int(tunit / timeStepSize) - 1)  # randomly select centerpoint time
        
        # Random bubble size - proportional to remaining gas needed
        mg_bubble = (random.random()) * (mg_needed - local_mg_defined)
        
        print_info(f"Trying bubble at CID {C_ID}, remaining gas: {mg_needed - local_mg_defined}")
        
        bubbleDefined, mg_checked, updated_UVal, updated_VOFwVal = bubbleShape(
            local_UVal, local_VOFwVal, coordList, timeVal, normalInlet, 
            timeStepSize, rhog, U, tunit, t, C_ID, C_t, t, 
            shapeID, mg_bubble, mg_needed - local_mg_defined
        )
        
        if bubbleDefined:
            print_info(f"Bubble placed: {mg_checked} kg")
            local_mg_defined += mg_checked
            local_UVal = updated_UVal
            local_VOFwVal = updated_VOFwVal
            successful_bubbles.append((C_ID, C_t, shapeID, mg_bubble, mg_checked))
            local_iter = 0
        else:
            local_iter += 1
            
    result_status = "success" if local_mg_defined >= (mg_needed - tol_mg) else "failed"
    print_info(f"Time interval {t} complete: {local_mg_defined} kg placed, status: {result_status}")
    
    return t, local_mg_defined, successful_bubbles, local_UVal, local_VOFwVal, result_status

def main():
    start_time = time.time()
    
    # Initialize UVal and VOFwVal once for all processes
    UVal = np.ones([len(coordList), nTimeSteps, 3])  # initially: "pre-inlet domain" at constant velocity
    for i in np.arange(3):
        UVal[:, :, 0] = U*normalInlet[0]
        UVal[:, :, 1] = U*normalInlet[1]
        UVal[:, :, 2] = U*normalInlet[2]
    VOFwVal = np.ones([len(coordList), nTimeSteps, 1])  # initially: "pre-inlet domain" filled with water
    
    # Prepare data for parallel processing
    nIntervals = int((endTime-startTime)/tunit)
    print(f"Between startTime {startTime}s and endTime {endTime}s, {nIntervals} intervals of {tunit}s will be processed in parallel.")
    
    # Shared data that all processes will use
    shared_data = (UVal, VOFwVal, coordList, timeVal, normalInlet)
    
    # Create task list - each task is one time interval
    tasks = [(t, mg_tunit, shared_data) for t in range(nIntervals)]
    
    # Determine number of cores to use (leave one core free for system)
    #num_cores = max(1, mp.cpu_count() - 1)
    num_cores = 2
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    # Execute in parallel
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(try_place_bubble, tasks)
    
    # Process results and merge the data
    for t, mg_defined, successful_bubbles, local_UVal, local_VOFwVal, status in results:
        print(f"Time interval {t} result: {mg_defined} kg of gas inserted. (desired: {mg_tunit} kg). Status: {status}")
        
        # Update master UVal and VOFwVal with results from this time interval
        time_slice = slice(t * int(tunit / timeStepSize), (t + 1) * int(tunit / timeStepSize))
        UVal[:, time_slice, :] = local_UVal[:, time_slice, :]
        VOFwVal[:, time_slice, :] = local_VOFwVal[:, time_slice, :]
        
        if status == "failed":
            print(f"WARNING: Time interval {t} did not reach target gas mass within iteration limit!")
    
    elapsed_time = time.time() - start_time
    print(f"Inlet was modelled successfully in {elapsed_time:.2f} seconds using {num_cores} cores.")
    

    # plot the bubble distrbution
    fig, ax = plt.subplots()
    X,Y,Z=coordList[:,1],coordList[:,3],VOFwVal[:,-8,0]
    ax.tricontour(X,Y,Z,levels=10,colors='k')
    cntr=ax.tricontourf(X,Y,Z,cmap="RdBu_r")
    fig.colorbar(cntr,ax=ax)
    ax.plot(X,Y,'ko',ms=3)
    ax.set(xlim=(-0.108,0.108),ylim=(0,D))
    ax.set_title(f'Bubble Inlet at timestep {endTime}')
    
    #plt.show()
    plt.savefig('inletDefinition-VOFw.png', dpi=100, bbox_inches='tight')
    
    # Here you could save the results to file or .csv
    # np.save("UVal_result.npy", UVal)
    # np.save("VOFwVal_result.npy", VOFwVal)
    print("Saving inlet profile to CSV-files. ")
    file = 'bubbles_time_001_vf_ck=0.2.csv'
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
    

if __name__ == "__main__":
    main()