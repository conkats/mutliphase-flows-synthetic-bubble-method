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
import argparse

def print_info(message):
    """Print with process info for debugging"""
    print(f"Process {mp.current_process().name}: {message}")

def create_rectangular_inlet(x_min=-0.108, x_max=0.108, z_min=0, z_max=0.06, nx=220, nz=20, y_value=-4.0800e-01):
    """
    Creates a rectangular grid inlet geometry.
    
    Parameters:
    -----------
    x_min, x_max : float
        Min and max values for x-coordinate
    z_min, z_max : float
        Min and max values for z-coordinate
    nx, nz : int
        Number of points in x and z directions
    y_value : float
        Fixed y-coordinate for the inlet plane
        
    Returns:
    --------
    coordList : ndarray
        Array with columns [ID, x, y, z, area]
    normal : ndarray
        Normal vector to the inlet
    """
    # Create coordinate arrays
    x = np.linspace(x_min, x_max, nx)
    z = np.linspace(z_min, z_max, nz)
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    
    # Create 2D mesh grid for x and z
    xv, zv = np.meshgrid(x, z)
    
    # Flatten the arrays
    xvc = xv.flatten()
    zvc = zv.flatten()
    
    # Create y-coordinates (all the same value)
    yvc = np.ones(len(xvc)) * y_value
    
    # Construct coordinate list with IDs and face areas
    coordList = np.array([
        np.arange(len(xvc)),  # ID
        xvc,                  # x-coordinate
        yvc,                  # y-coordinate
        zvc,                  # z-coordinate
        np.ones(len(xvc)) * dx * dz  # face area
    ]).T
    
    # Normal vector points in positive y direction
    normal = np.array([0, 1, 0])
    
    print(f"Rectangular inlet created with {len(xvc)} points")
    print(f"Grid dimensions: {nx} x {nz}")
    print(f"Grid spacing: dx={dx:.6f}, dz={dz:.6f}")
    print(f"Total inlet area: {np.sum(coordList[:, 4]):.6f}")
    
    return coordList, normal

def create_circular_inlet(radius=0.0105, nxy=40, y_value=0.0, normal_direction="z"):
    """
    Creates a circular inlet geometry.
    
    Parameters:
    -----------
    radius : float
        Radius of the circular inlet
    nxy : int
        Number of points in both x and y directions in the square grid
        (will be filtered to keep only points inside the circle)
    y_value : float
        Fixed coordinate for the inlet plane
    normal_direction : str
        Direction of the normal vector ("x", "y", or "z")
        
    Returns:
    --------
    coordList : ndarray
        Array with columns [ID, x, y, z, area]
    normal : ndarray
        Normal vector to the inlet
    """
    # Create coordinate arrays for a square that will contain the circle
    x = np.linspace(-radius, radius, nxy)
    y = np.linspace(-radius, radius, nxy)
    dxy = x[1] - x[0]
    
    # Create 2D mesh grid for x and y
    xv, yv = np.meshgrid(x, y)
    
    # Create mask for points inside the circle
    criterion = np.array(xv*xv + yv*yv < radius*radius)
    
    # Filter points to keep only those inside the circle
    xvc = xv[criterion]
    yvc = yv[criterion]
    
    # Create the third coordinate (all the same value)
    if normal_direction == "x":
        zvc = np.zeros(xvc.shape)
        coords = np.array([np.ones(len(xvc)) * y_value, yvc, zvc])
        normal = np.array([1, 0, 0])
    elif normal_direction == "y":
        zvc = np.zeros(xvc.shape)
        coords = np.array([xvc, np.ones(len(xvc)) * y_value, zvc])
        normal = np.array([0, 1, 0])
    else:  # z direction
        coords = np.array([xvc, yvc, np.ones(len(xvc)) * y_value])
        normal = np.array([0, 0, 1])
    
    # Construct coordinate list with IDs and face areas
    coordList = np.array([
        np.arange(len(xvc)),  # ID
        coords[0],            # x-coordinate
        coords[1],            # y-coordinate
        coords[2],            # z-coordinate
        np.ones(len(xvc)) * dxy * dxy  # face area
    ]).T
    
    print(f"Circular inlet created with {len(xvc)} points")
    print(f"Grid dimensions: {nxy} x {nxy} (filtered to circle with radius {radius})")
    print(f"Grid spacing: dxy={dxy:.6f}")
    print(f"Total inlet area: {np.sum(coordList[:, 4]):.6f}")
    print(f"Theoretical circle area: {np.pi * radius * radius:.6f}")
    
    return coordList, normal

def plot_inlet_geometry(coordList, title="Inlet Geometry", show=True):
    """Plots the inlet geometry for visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Determine which columns to plot based on non-zero variance
    cols = [1, 2, 3]  # x, y, z columns
    variances = [np.var(coordList[:, col]) for col in cols]
    
    # Find the two columns with highest variance
    plot_cols = sorted(range(len(variances)), key=lambda i: variances[i], reverse=True)[:2]
    plot_cols = [cols[i] for i in plot_cols]
    
    # Default column names
    col_names = ['x', 'y', 'z']
    
    # Plot points
    scatter = ax.scatter(
        coordList[:, plot_cols[0]], 
        coordList[:, plot_cols[1]], 
        s=5,  # point size
        c=coordList[:, 4],  # color based on area
        cmap='viridis',
        alpha=0.7
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Face Area')
    
    # Set labels
    ax.set_xlabel(col_names[plot_cols[0] - 1])
    ax.set_ylabel(col_names[plot_cols[1] - 1])
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add stats to plot
    stats_text = (
        f"Total points: {len(coordList)}\n"
        f"Total area: {np.sum(coordList[:, 4]):.6f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig, ax

def bubbleShape(UVal, VOFwVal, coordList, timeVal, normalInlet, timeStepSize, rhog, U, tunit, t, C_ID, C_t, timeInterval, shapeID, mgb, mg_StillRequired):
    """
    Define bubble shape at a given point and time
    Now accepts UVal and VOFwVal as parameters instead of using globals
    """
    UVal_temp = np.array(UVal)
    VOFwVal_temp = np.array(VOFwVal)
    
    C_coord = coordList[C_ID, :]
    C_time = timeVal[C_t]
    if shapeID > 0:  # Assuming only one shape (0) is defined
        print_info(f"Fatal error. Shape generator asks for non-existing bubble shape.")
        return False, 0.0, UVal, VOFwVal
       
    # Shape ID 0 - Spherical bubble
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
        
    # Check that the center point meets requirements
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
        shapeID = 0  # Only one bubble shape defined
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
    global startTime, endTime, timeStepSize, tunit, rhog, U, intersectBoundary, intersectBubble, mg_tunit, tol_mg, n_max
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Inlet modeling with bubble generation')
    parser.add_argument('--geometry', type=str, default='rectangular', choices=['rectangular', 'circular'],
                      help='Type of inlet geometry (rectangular or circular)')
    parser.add_argument('--startTime', type=float, default=0.0, 
                      help='Start time for simulation')
    parser.add_argument('--endTime', type=float, default=1.0, 
                      help='End time for simulation')
    parser.add_argument('--timeStepSize', type=float, default=0.01, 
                      help='Time step size')
    parser.add_argument('--tunit', type=float, default=0.1, 
                      help='Time unit for bubble generation')
    parser.add_argument('--U', type=float, default=0.463, 
                      help='Velocity of the mixture')
    parser.add_argument('--v_ratio', type=float, default=0.2, 
                      help='Void ratio')
    parser.add_argument('--D', type=float, default=0.2, 
                      help='Depth of spanwise')
    parser.add_argument('--n_max', type=int, default=1000, 
                      help='Maximum number of iterations for bubble placement')
    parser.add_argument('--verbose', type=int, default=0, 
                      help='Verbosity level')
    parser.add_argument('--ncpu', type=int, default=0, 
                      help='Number of cores')
    parser.add_argument('--save', type=str, default=None, 
                      help='Save results to the specified prefix (e.g., "results" saves to results_UVal.npy and results_VOFwVal.npy)')
    parser.add_argument('--plot', action='store_true', 
                      help='Plot inlet geometry')
    
    args = parser.parse_args()
    
    # Set global parameters
    startTime = args.startTime
    endTime = args.endTime
    timeStepSize = args.timeStepSize
    tunit = args.tunit
    U = args.U
    v_ratio = args.v_ratio
    D = args.D
    ncpu = args.ncpu
    n_max = args.n_max
    verbose = args.verbose
    
    # Fixed parameters
    rhog = 1.18415  # Density of the gas
    rhol = 997.561  # Density of the liquid
    intersectBoundary = str(True)
    intersectBubble = str(True)
    
    # Validate time parameters
    if int((endTime-startTime)/tunit) != ((endTime-startTime)/tunit):
        sys.exit("The desired time interval (endTime - startTime) should be a multiple of tunit.")
    if endTime <= startTime:
        sys.exit("The endTime should be larger than the startTime.")
    if (abs(int(tunit/timeStepSize) - tunit/timeStepSize) >= timeStepSize) and \
       (abs((int(tunit/timeStepSize)+1) - tunit/timeStepSize) >= timeStepSize):
        sys.exit("Variable tunit should be a multiple of timeStepSize.")
    
    # Create inlet geometry based on command line option
    if args.geometry == 'rectangular':
        coordList, normalInlet = create_rectangular_inlet()
    else:  # circular
        coordList, normalInlet = create_circular_inlet(radius=0.0105, nxy=40)
    
    # Calculate inlet area and mass flow rates
    total_area = np.sum(coordList[:, 4])
    mg_tunit = float(total_area * v_ratio * rhog * U) * tunit
    tol_mg = float(1e-5)
    
    print(f"Total inlet area: {total_area:.6f}")
    print(f"Gas mass per tunit: {mg_tunit:.6f} kg")
    
    # Plot inlet geometry if requested
    if args.plot:
        plot_inlet_geometry(coordList, f"{args.geometry.capitalize()} Inlet Geometry")
    
    # Initialize simulation
    start_time = time.time()
    
    # Initialize UVal and VOFwVal once for all processes
    nTimeSteps = int((endTime-startTime)/timeStepSize) + 1
    timeVal = np.arange(startTime, endTime, timeStepSize)
    
    UVal = np.ones([len(coordList), nTimeSteps, 3])  # initially: "pre-inlet domain" at constant velocity
    for i in np.arange(3):
        UVal[:, :, 0] = U * normalInlet[0]
        UVal[:, :, 1] = U *normalInlet[1]
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
    num_cores = ncpu#2
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