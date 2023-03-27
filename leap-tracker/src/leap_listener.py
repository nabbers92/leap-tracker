import sys
sys.path.insert(0, "../lib/")
import Leap, thread, time, inspect

class HandListener(Leap.Listener):

    def on_connect(self, controller):
        print "dt,x,z,vx,vz"

    def on_frame(self, controller):
        # frame indexing is 0 for current frame, 1 for previous frame, etc.
        prev_frame = controller.frame(1)
        current_frame = controller.frame(0)

        # getting positions of first hand index finger for  current and previous frames
        curr_position = current_frame.hands[0].fingers[1].bone(3).next_joint
        prev_position = prev_frame.hands[0].fingers[1].bone(3).next_joint

        # distances reported as mm
        dx = curr_position - prev_position

        # timestamp reported as time since Leap Motion Controller connected,
        # changed from microseconds to seconds
        dt = (float(current_frame.timestamp) - float(prev_frame.timestamp))/(10**6)

        # in camera coordinate system, we want the x and z elements
        print '{}, {}, {}, {}, {},'.format(
            dt, curr_position[0], curr_position[2], (dx/dt)[0], (dx/dt)[2])
        
    def on_disconnect(self, controller):
        print "Disconnected"


