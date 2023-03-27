import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy.stats import skewnorm

def generate_tuning_curves(base_curves, n_units, seed):
    """
    Creates a series of neural tuning curves by translating and scaling a set of
    given base curves.

    Args:
        base_curves (array, n_curves x 360): The base tuning curve used as a 
            template for the generated tuning curves
        n_units (int): The number of desired tuning curves
        seed (int): Randomization seed used in uniform distribution sampling and 
            curve shuffling

    Returns:
        tuning_curves (array, n_units X 360): Each row of the matrix contains a 
            tuning curve that has been shifted and scaled
    """

    np.random.seed(seed)
    n_curves = base_curves.shape[0]
    tuning_curves = np.zeros((n_units, 360))

    for n in range(n_units):
        base_curve_idx = np.random.randint(0, n_curves)
        base_curve = base_curves[base_curve_idx,:]
        tuning_curves[n,:] = np.roll(base_curve, (2*n)*(360//n_units))
        tuning_curves[n,:] = tuning_curves[n,:] * np.random.uniform(low=0.15, 
                                                                    high=1.0, 
                                                                    size=None)

    np.random.shuffle(tuning_curves)
    
    return tuning_curves

def generate_spikes(tuning_curves, dt, vx, vz, seed):
    """
    Generates neural spike data at a timestep based off of direction of movement
    and the firing rate of neurons in tuning_curves. Spikes are returned as a 1,
    while non-firing is marked by a 0. Note: xz-plane used because of LeapMotion 
    coordinate axes conventions

    Args:
        tuning_curves (array, n_units X 360): Tuning curves that determine a 
            neuron's firing rate given a direction of movement
        dt (float): Timestep increment, given in seconds
        vx (float): The vector of motion on the x-axis of the xz-plane
        vz (float): The vector of motion on the z-axis of the xz-plane
        seed (int): Randomization seed used in uniform distribution sampling   

    Returns:
        spikes (array, n_units X 1): array returning firing/nonfiring state of 
            a neuron at the given timepoint.
    """
    
    np.random.seed(seed)
    n_units = tuning_curves.shape[0]
    angle = np.rad2deg(np.arctan2(vx, vz)[0])
    angle = np.rint(angle).astype(int)

    angle = angle % 360
    skew = 10
    if np.sqrt(vx**2 + vz**2) < 10:
        fr = skewnorm.rvs(skew, 
                          scale=2, 
                          size=tuning_curves[:,angle].shape).astype(int)+2
    else:
        fr = tuning_curves[:, angle]

    spikes = np.random.uniform(0,1, n_units) < (fr*dt)/4
    spikes = spikes.reshape((-1, 1)).astype(int)

    return spikes

def create_spikes_array(data_array, tuning_curves, sigma, seed):
    """
    Creates a raster plot of the generated neural spiking data across the 
        duration of data.

    Args:
        data_array (array, N X 5, N = duration of data): Array of measurements 
            taken by the leapMotion tracker including timestep, x position, 
            z position, dx, dz
        tuning_curves (array, n_units X 360): Tuning curves that determine a 
            neuron's firing rate given a direction of movement
        sigma (float): standard deviation to be used in convolution with the
            gaussian kernel
        seed (int): Randomization seed used to generate seeds for individual 
            timestep spike generation   

    Returns:
        spikes_array (array, N X n_units): spiking data for duration of 
            recording
        smoothed_spikes (array, N X n_units): spiking data convolved with a 
            gaussian kernel with standard deviation sigma
        time (array, 1 X N) time points for recording, aligned to "go" cue

    """
    
    np.random.seed(seed)

    dt = data_array[:,0].reshape(-1,1)
    x = data_array[:,1].reshape(-1,1)
    z = data_array[:,2].reshape(-1,1)
    vx = data_array[:,3].reshape(-1,1)
    vz = data_array[:,4].reshape(-1,1)
    
    n_units = tuning_curves.shape[0]
    spikes_array = np.zeros((n_units, dt.shape[0]))
    time = np.zeros((dt.shape))

    for n in range(dt.shape[0]):
        seed_num = np.random.randint(0, 2**32 - 1)
        spikes = generate_spikes(tuning_curves, dt[n], vx[n], vz[n], seed_num)
        spikes_array[:,n] = spikes[:,0]
        if n > 0:
            time[n] = time[n-1] + dt[n]

    smoothed_spikes = scipy.ndimage.gaussian_filter1d(spikes_array, sigma, 
                                                      axis=1)

    return spikes_array, smoothed_spikes, time

def normalize_block(block_array, spike_array, tuning_curves, sigma, seed):
    spikes_block, smoothed_block, time = create_spikes_array(block_array, tuning_curves, sigma, seed)
    mu = np.mean(spikes_block)
    sigma_var = np.std(spikes_block)
    normalized_spikes = spike_array - mu
    normalized_spikes /= sigma_var
    smoothed_normalized = scipy.ndimage.gaussian_filter1d(normalized_spikes, sigma, axis=1)

    return normalized_spikes, smoothed_normalized

def create_spikes_array_pandas(filename, tuning_curves, seed):
    """
    Creates a raster plot of the generated neural spiking data across the 
        duration of data.

    Args:
        time (array, 1 X N, N = duration of data): Array of timepoints in data 
            collection, incremented by dt
        spike_data (array, n_units X N): Array with each row containing the 
            spiking data for a neuron
        plot_title (any): Title to be given to the raster plot
        seed (int): Randomization seed used to generate seeds for individual 
            timestep spike generation   

    Returns:
        Raster plot of neural firing data
    """
    
    np.random.seed(seed)
    df = pd.read_csv(filename, index_col=False)
    df = df.drop(labels=0, axis=0, inplace=False)
    df = df.dropna()
    df = df.iloc[:-1]
    dt = np.array(df.dt).reshape(-1,1).astype(float)
    z = np.array(df.x).reshape(-1,1).astype(float)
    x = np.array(df.z).reshape(-1,1).astype(float)
    vz = np.array(df.vx).reshape(-1,1).astype(float)
    vx = np.array(df.vz).reshape(-1,1).astype(float)
    
    n_units = tuning_curves.shape[0]
    spikes_array = np.zeros((n_units, dt.shape[0]))
    time = np.zeros((dt.shape))

    for n in range(dt.shape[0]):
        seed_num = np.random.randint(0, 2**32 - 1)
        spikes = generate_spikes(tuning_curves, dt[n], vx[n], vz[n], seed_num)
        spikes_array[:,n] = spikes[:,0]
        if n > 0:
            time[n] = time[n-1] + dt[n]

    smoothed_spikes = scipy.ndimage.gaussian_filter1d(spikes_array, 3.0, axis=1)

    return spikes_array, smoothed_spikes, time, dt, z, x, vz, vx

def plot_raster(time, cue, spike_data, plot_title):
    """
    Creates a raster plot of the generated neural spiking data across the 
        duration of data.

    Args:
        time (array, 1 X N, N = duration of data): Array of timepoints in data 
            collection, incremented by dt
        cue (float): Time at which the "go" cue was given to begin movement
        spike_data (array, n_units X N): Array with each row containing the 
            spiking data for a neuron
        plot_title(any): Title to be given to the raster plot    

    Returns:
        Raster plot of neural firing data
    """

    time -= cue
    n_units = spike_data.shape[0]
    for n in range(n_units):
        spike_data_n = spike_data[n,:].astype(int)
        spike_positions = time[spike_data_n != 0]

        for m in range(spike_positions.shape[0]):
            plt.plot([spike_positions[m], spike_positions[m]],
            [n-(n_units/750), n+(n_units/750)], color='black')

    plt.plot([0, 0], [0, n_units], color='red')
    plt.xlabel('Time (s)')
    # plt.xlim([0, time[-1]])
    plt.ylabel('Neuron (#)')
    plt.ylim([0-n_units/100, n_units + n_units/750])
    plt.title('Raster Plot: {}'.format(plot_title))


def generate_synthetic_data(data, tuning_curves, vx_scalers, vz_scalers, seed):
    """
    Creates synthetic data for a center-out task. Synthetic data is generated by
    multiplying the components of the x- and z-vectors by small floating-point
    numbers between 0.8 and 1.2.

    Args:
        data (dict): dictionary containing numpy arrays of the firing rates of  
            the first 0.6 seconds of movement during the center-out task. The 
            key-value pairs are direction (str): firing rate array
        tuning_curves (array, n_units X len(dt)): array with each row containing
            a tuning curve for each neuron in the model.
        vx_scalers (array): array containing the values to multiply with vx 
            during synthetic data generation; works best if elements are between
            0.8 and 1.2
        vz_scalers (array): array containing the values to multiply with vz 
            during synthetic data generation; works best if elements are between
            0.8 and 1.2

    Returns:
        data (dict): dictionary containing numpy arrays of synthetic data 
            generated from the original measured data. Adds a third dimension to
            the arrays increasing the batch size for each direction.
    """

    np.random.seed(seed)
    dt = data[:,0].reshape(-1,1).astype(float) 
    x = data[:,2].reshape(-1,1).astype(float)
    z = data[:,3].reshape(-1,1).astype(float)
    vx = data[:,4].reshape(-1,1).astype(float)
    vz = data[:,5].reshape(-1,1).astype(float)

    n_units = tuning_curves.shape[0]
    k = 0
    for num1 in vx_scalers:
        for num2 in vz_scalers:
            time = np.zeros((dt.shape))
            spikes_array = np.zeros((n_units, dt.shape[0]))
            for m in range(dt.shape[0]):
                seed_num = np.random.randint(0, 2**32 - 1)
                spikes = generate_spikes(tuning_curves, dt[m], 
                                         vx[m]*num1, vz[m]*num2, seed_num)
                spikes_array[:,m] = spikes[:,0]
                if m > 0:
                    time[m] = time[m-1] + dt[m]
            temp_data = np.concatenate((dt, time, x, z, vx*num1, 
                                        vz*num2, spikes_array.T), axis=1)
            if k == 0:
                data = np.concatenate([data[..., None], 
                                       temp_data[..., None]], axis=2)   
            else:
                data = np.concatenate([data, temp_data[..., None]], axis=2)
            k += 1
    print(k)
    return data
