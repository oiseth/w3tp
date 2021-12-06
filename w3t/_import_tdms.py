

__all__ = ["tdms2h5_4loadcells", ]

def tdms2h5_4loadcells(h5file,tdmsfile):
    """ imports data from tdms
        
    Arguments
        ----------
        h5file      : hdf5 filename 
        tdmsfile    : height     
        ----------      
    
    Imports data from wind tunnel experiment stored in the *.tdms file, 
    converts the data using calibration coefficients and stores the processed data as a 
    group in the hdf5 file 
    
    """
    import numpy as np
    from scipy import linalg as spla
    from nptdms import TdmsFile
    import h5py
    
    # Read *.tdms file
    tdms_file = TdmsFile.read(tdmsfile)
    
    # Properties
    samplig_frequency = float(tdms_file.properties['04  Logging Frequency'][0:3])
    author = tdms_file.properties['Author']
    c_coeff = float(tdms_file.properties['06 C Coefficient'][0:12].replace(',','.'))
    air_density = float(tdms_file.properties['08 Air density'][0:12].replace(',','.'))
    motion = tdms_file.properties['09 Motion path']
    temperature = float(tdms_file.properties['10 Temperature'][0:12].replace(',','.'))
    n_samples = tdms_file.groups()[1]['Load_1_1'].data.shape[0] # Number of elements in data array
    
    #Displacements
    volt2disp = np.array([390/36/1000, 390/36/1000, 40*2*np.pi/360])
    um = np.zeros((n_samples,3))
    um[:,0] = tdms_file.groups()[1]['Position_Horizontal'].data*volt2disp[0]
    um[:,1] = tdms_file.groups()[1]['Position_Vertical'].data*volt2disp[0]
    um[:,2] = tdms_file.groups()[1]['Position_Torsion'].data*volt2disp[0]
    transform_local2global = np.array([[-1, 0, 0],[ 0, 1, 0], [0, 0, -1]]) # Transformation matrix from local to global coordinates
    um = (transform_local2global @ um.T).T 
    
    # Wind velocity
    pressure_volt = tdms_file.groups()[1]['Wind_Speed'].data
    wind_velocity = (np.abs(2*c_coeff*pressure_volt/air_density))**0.5
    
    # Forces
    forces_volt = np.zeros((n_samples,24)) # Volt signal from load cells (4 load cells x 6 signals)
    pos = -1
    for k1 in range(4):
        for k2 in range(6):
            pos = pos + 1
            forces_volt[:,pos] = tdms_file.groups()[1]['Load_' + str(k1+1) + '_' + str(k2+1)].data
    
    # Calibration matrices for load cells. Transfroms from volts to forces and moments
    # Road side of the wind tunnel (load cell 1)
    ft16754 = np.array([[0.031162, 0.022256, 0.426060, -13.727634, -0.453720, 14.142106],
            [-0.181994, 16.057936, 0.280978, -7.925364, 0.131318, -8.198253],
            [24.791092, -1.301293, 25.152141, -1.654921, 24.971345, -1.281474],
            [0.000339, 0.194547, -0.722266, -0.045622, 0.721243, -0.139443],
            [0.829859, -0.046837, -0.420671, 0.194034, -0.408387, -0.147955],
            [0.007045, -0.430021, 0.008493, -0.426733, 0.013391, -0.441705]])
    
    # Front side of the wind tunnel (load cell 2)
    ft16752 = np.array([[0.138943, 0.025859, -0.070779, -14.040129, -0.150047, 14.158609],
            [-0.151285, 15.705065, -0.017973, -8.088157, 0.064626, -8.211768],
            [24.913282, -0.977383, 25.063880, -1.166585, 25.089508, -1.281523],
            [-0.001766, 0.190874, -0.724075, -0.062573, 0.724010, -0.138939],
            [0.833128, -0.033786, -0.412226, 0.190444, -0.412299, -0.150806],
            [0.004158, -0.422383, 0.002291, -0.436720, -0.002173, -0.439667]]);
    # Road side of the wind tunnel (load cell 3)
    ft25129 = np.array([[-0.23690,  -0.02985,   0.83228, -13.45095,  -0.44614,  14.15611],
            [-0.66311,  16.22499,   0.30221,  -7.81834,   0.46729,  -8.22419],
            [24.81924,  -0.86339,  25.25221,  -0.74013,  24.79589,  -1.12238],
            [-0.01080,   0.19534,  -0.71863,  -0.07687,   0.71982,  -0.12673],
            [0.83774,  -0.02489,  -0.42591,   0.17305,  -0.40583,  -0.15392],
            [0.01530,  -0.43881,   0.02286,  -0.42298,   0.02000,  -0.44578]])
    # Front side of the wind tunnel (load cell 4)
    ft25127 = np.array([[ -0.17520,  -0.04339,  -0.06436, -13.87662,  -0.30076,  14.06334],
            [-0.00197,  16.32323,  -0.21631,  -8.07687,   0.31384,  -8.18615],
            [24.70574,  -0.96329,  25.35344,  -1.20373,  24.58101,  -0.83064],
            [-0.00675,   0.19648,  -0.72452,  -0.06769,   0.71387,  -0.11683],
            [0.83273, -0.02797,  -0.41884,   0.18405,  -0.40480,  -0.15655],
            [-0.00027,  -0.44063,  -0.00540,  -0.44258,   0.01285,  -0.44384]])
        
    # Transform voltage signals to local corotated load cell axis
    transform_volt2local_corotated_forces = spla.block_diag(ft16754,ft16752,ft25129,ft25127)
    
    # Transform orientation of the load cells
    transform_local2global_ft16754 = np.array([[1, 0, 0],[0, 0, -1],[0, 1, 0]]) #(Road side)
    transform_local2global_ft16752 = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])    #(Front side)
    transform_local2global_ft25129 = np.array([[1, 0, 0], [0, 0, -1],[ 0, 1, 0]]) # (Road side)
    transform_local2global_ft25127 = np.array([[-1, 0, 0], [0, 0, 1],  [0, 1, 0]]) # (Front side)
    transform_Local2Global_load_cells = spla.block_diag(transform_local2global_ft16754,
                                                        transform_local2global_ft16754,
                                                        transform_local2global_ft16752,
                                                        transform_local2global_ft16752,
                                                        transform_local2global_ft25129,
                                                        transform_local2global_ft25129,
                                                        transform_local2global_ft25127,
                                                        transform_local2global_ft25127)
    local_corotated_forces = (transform_Local2Global_load_cells @ transform_volt2local_corotated_forces  @ forces_volt.T).T
    
    # Transform from eccentric forces and moments to rotation axis
    transfrom_center2eccentric1 = np.eye(6)
    ey1 = 0 
    ez1 = 0
    ex1 = -0.1
    ex2 = 0.1
    
    transfrom_center2eccentric1[0,4] =  ez1 
    transfrom_center2eccentric1[0,5] = -ey1
    transfrom_center2eccentric1[1,3] = -ez1 
    transfrom_center2eccentric1[1,5] = ex1
    transfrom_center2eccentric1[2,3] = ey1 
    transfrom_center2eccentric1[2,4] = -ex1
    
    transfrom_center2eccentric2 = np.eye(6)
    
    ey2 = 0 
    ez2 = 0
    
    transfrom_center2eccentric2[0,4] =  ez2
    transfrom_center2eccentric2[0,5] = -ey2
    
    transfrom_center2eccentric2[1,3] = -ez2 
    transfrom_center2eccentric2[1,5] = ex2;
    
    transfrom_center2eccentric2[2,3] = ey2 
    transfrom_center2eccentric2[2,4] = -ex2;
    transfrom_center2eccentric = spla.block_diag(transfrom_center2eccentric1,transfrom_center2eccentric1,transfrom_center2eccentric2,transfrom_center2eccentric2)
    
    #Forces in fixed global coordinates
    local_fixed_forces = np.zeros(local_corotated_forces.shape)
    center_fixed_forces = np.zeros(local_corotated_forces.shape)
    angles = um[:,2]-np.mean(um[0:200])
    for k  in range(forces_volt.shape[0]):
        transfrom_corotated2fixed_n = np.array([[np.cos(angles[k]), 0, np.sin(angles[k])],[0, 1, 0], [-np.sin(angles[k]), 0, np.cos(angles[k])]]);
        transfrom_corotated2fixed = spla.block_diag(transfrom_corotated2fixed_n,transfrom_corotated2fixed_n,transfrom_corotated2fixed_n,transfrom_corotated2fixed_n,transfrom_corotated2fixed_n,transfrom_corotated2fixed_n,transfrom_corotated2fixed_n,transfrom_corotated2fixed_n)
        local_fixed_forces[k,:] = (transfrom_corotated2fixed @ local_corotated_forces[k,:].T).T;
        center_fixed_forces[k,:] = (transfrom_corotated2fixed @ transfrom_center2eccentric.T @ local_corotated_forces[k,:].T).T
    
    #%% save data to hdf5 file
    # Open file
    f = h5py.File((h5file + ".hdf5"), "a")
    f.attrs["project description"] = "Forced vibration wind tunnel tests conducted at NTNU" 
    
    # Create group for experiment 
    grp = f.create_group(tdmsfile[str.rfind(tdmsfile,'\\')+1:-5])
    grp.attrs['test_type'] = "quasi static test"
    grp.attrs["sampling frequency"] = samplig_frequency
    grp.attrs["test operator"] = author
    grp.attrs["forced motion"] = motion
    
    # Create temperature dataset 
    dataset_temperature = grp.create_dataset("temperature", data = temperature)
    dataset_temperature.attrs["description"] = "temparature in the wind tunnel measured by termocouple"
    dataset_temperature.attrs["units"] = ["C"]
    
    # create wind dataset
    dataset_wind_velocity = grp.create_dataset('wind_velocity',data=wind_velocity, dtype = "float64")
    dataset_wind_velocity.attrs['description'] = "Wind velocity measured by pitot tube"
    dataset_wind_velocity.attrs['units'] = ["m/s"]
    
    # create motion dataset
    dataset_motion = grp.create_dataset('u',data=um, dtype ="float64")
    dataset_motion.attrs['description'] = "Motion measured by the encoders on the servo motors in global coordinates"
    dataset_motion.attrs['units'] = ["m", "m", "rad"]
    
    # create measured forces dataset
    dataset_fg = grp.create_dataset('fg',data=local_fixed_forces, dtype = "float64")
    dataset_fg.attrs['description'] = "Forces in global coordinates for all load cells"
    dataset_fg.attrs['units'] = ["N", "N", "N", "Nm", "Nm", "Nm","N", "N", "N", "Nm", "Nm", "Nm","N", "N", "N", "Nm", "Nm", "Nm","N", "N", "N", "Nm", "Nm", "Nm"]
    
    # create measured forces transformed to center of rotation dataset 
    dataset_fgc = grp.create_dataset('fgc',data=center_fixed_forces, dtype = "float64")
    dataset_fgc.attrs['description'] = "Forces in global coordinates for all load cells"
    dataset_fgc.attrs['units'] = ["N", "N", "N", "Nm", "Nm", "Nm","N", "N", "N", "Nm", "Nm", "Nm","N", "N", "N", "Nm", "Nm", "Nm","N", "N", "N", "Nm", "Nm", "Nm"]
    dataset_fgc.attrs['eccentricity'] = np.array([ex1, ex2]) 
        
    f.close()