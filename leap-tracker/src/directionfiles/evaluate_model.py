import os
import pickle
import numpy as np

np.random.seed(1000)

base_dir = os.path.expanduser('~') + '/leapTracker/datafiles/direction'

test_filename = os.path.join(base_dir, 'final_testing_data.pkl')
test_data = pickle.load(open(test_filename, 'rb'))

model_path = os.path.join(base_dir, 'lda_classifier.mdl')
model = pickle.load(open(model_path, 'rb'))

X = test_data['firing']
y = test_data['directions']

new_data = np.concatenate((X,y.reshape(-1,1)), axis=1)
np.random.shuffle(new_data)

X = new_data[:,0:50]
y = new_data[:,-1]

percentage = 0
for i in range(X.shape[0]):
    yhat = model.predict(X[i].reshape(1,-1))
    print('Predicted Direction: {} Degrees'.format(int(yhat[0])))
    print('Actual Direction: {} Degrees'.format(int(y[i])))
    if int(yhat[0]) == int(y[i]):
        percentage += 1
print('Accuracy: {}%'.format((percentage/X.shape[0])*100))