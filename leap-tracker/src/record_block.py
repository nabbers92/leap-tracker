import time
import os
import argparse
import subprocess
import numpy as np
import time
import pickle
from datetime import date

parser = argparse.ArgumentParser(description='A program to record hand trajectories')
parser.add_argument('-t', '--trial', type=int)
args = parser.parse_args()
np.random.seed(10050)

chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '>', ',',
"'", '~', '?']

trial_num = args.trial
savedir = os.path.expanduser('~') + '/leapTracker/datafiles/letters/characterBlocks'
datafile = '{}/block{}_{}.csv'.format(savedir, str(trial_num), str(date.today()))
timingfile = '{}/block{}_{}.pkl'.format(savedir, str(trial_num), str(date.today()))

data = dict()
chars_prompt = []
idxs = np.arange(0, len(chars), 1)
np.random.shuffle(idxs)
timelist = np.array([])

print('Move hand over LeapMotion Tracker')

time.sleep(2)

with open(datafile, 'wb') as outfile:
    print('Beginning motion capture')
    p1 = subprocess.Popen(['python2', 'capture_motion_data.py'], stdout=outfile)
    time.sleep(2)
    subprocess.run(['clear'])
    timelist = np.append(timelist, time.time())

    for idx in idxs[0:1]:
        prompt_delay = np.random.exponential(scale=2.5)

        while prompt_delay < 2 or prompt_delay > 3:
            prompt_delay = np.random.exponential(scale=2.5)

        if idx == idxs[0]:
            print('First Character: {}'.format(chars[idx]))
        else:
            print('Next Character: {}'.format(chars[idx]))

        chars_prompt.append(chars[idx])
        time.sleep(prompt_delay)

        print('Begin')
        timelist = np.append(timelist, time.time())
        time.sleep(3)

        if idx != idxs[-1]:
            print('Reset')
            time.sleep(2)

    subprocess.run(['clear'])
    p1.kill()

print('All done, thank you!')

data['timelist'] = timelist
data['charPrompts'] = chars_prompt

f = open(timingfile, 'wb')
pickle.dump(data, f)
f.close()