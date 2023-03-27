import time
import argparse
import subprocess

parser = argparse.ArgumentParser(description='A program to record hand trajectories')
parser.add_argument('-n', '--number', type=int)
args = parser.parse_args()

chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '>', ',',
"'", '~', '?']

for i in range(len(chars)):
    print('Current Character: {}'.format(chars[i]))
    time.sleep(1)
    for j in range(args.number):
        print('Trial {}'.format(j+1))
        filename = '../datafiles/letters/character_{}_trial{}.csv'.format(i, j+1)
        time.sleep(1)
        with open(filename, 'w') as outfile:
            p1 = subprocess.Popen(['python2', 'capture_motion_data.py'], 
                                  stdout=outfile)
            time.sleep(0.5)
            print('Begin')
            time.sleep(1.5)
            p1.kill()
        print('Reset')
        time.sleep(2)

print('All done, thank you!')
