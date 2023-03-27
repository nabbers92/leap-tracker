import numpy as np
from joblib import Parallel, delayed
import scipy
import os
import multiprocessing
import pickle
# np.random.seed(1000)
seed = 10050
n_units = 50

from neural_tuning import (generate_tuning_curves, generate_spikes, plot_raster,
    generate_synthetic_data)

base_dir = os.path.expanduser('~') + '/leapTracker/datafiles/direction'
# test_dir = os.path.join(base_dir, 'training')
# train_dir = os.path.join(base_dir, 'testing')

tuning_data = scipy.io.loadmat(os.path.expanduser('~') + '/leapTracker/datafiles/tuning.mat')
tuning_data = np.array(tuning_data['tuningMat'])
tuning_curves = generate_tuning_curves(tuning_data, n_units, seed)

training_datafile_path = os.path.join(base_dir, 'raw_training_datafiles.pkl')
training_datafiles = pickle.load(open(training_datafile_path, 'rb'))

testing_datafile_path = os.path.join(base_dir, 'raw_test_datafiles.pkl')
testing_datafiles = pickle.load(open(testing_datafile_path, 'rb'))

labels = ['0', '45', '90', '135', '180', '225', '270', '315']
training_data = dict()
vx_scalers = np.random.uniform(0.8, 1.2, 5)
vz_scalers = np.random.uniform(0.8, 1.2, 5)

num_cores = multiprocessing.cpu_count()
training_data = Parallel(n_jobs=num_cores)(delayed(generate_synthetic_data)
                                                 (training_datafiles[label], 
                                                  tuning_curves, vx_scalers, 
                                                  vz_scalers, seed) for label in labels)

vx_scalers = [1]
vz_scalers = [1]

testing_data = Parallel(n_jobs=num_cores)(delayed(generate_synthetic_data)
                                                 (testing_datafiles[label], 
                                                  tuning_curves, vx_scalers, 
                                                  vz_scalers, seed) for label in labels)

print('Done augmenting data, now saving...')

f = open(os.path.join(base_dir, 'intermediate_training_datafiles.pkl'), 'wb')
pickle.dump(training_data, f)
f.close()

f = open(os.path.join(base_dir, 'intermediate_testing_datafiles.pkl'), 'wb')
pickle.dump(testing_data, f)
f.close()

print('Finished!')

