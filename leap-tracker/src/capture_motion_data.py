import sys
sys.path.insert(0, "../lib/")
import Leap, thread, time, inspect
from leap_listener import HandListener
import random

def main():
    
    # Create listener and controller
    listener = HandListener()
    controller = Leap.Controller()

    # Have listener receive events from controller
    controller.add_listener(listener)

    # Keep this process running until Enter is pressed
    # print "Press Enter to quit..." 
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        # Remove listener when done
        controller.remove_listener(listener)

if __name__ == "__main__":
    main()
    