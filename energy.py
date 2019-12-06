import time, subprocess, shlex, numpy as np
from datetime import datetime, timedelta
from copy import deepcopy as dcpy

# Creates an nvidia-smi process that logs our energy metrics (can be killed at-will)
def energyProfile(duration):
  # Create log of output using nvidia_smi
    # shlex.split() helps Popen() read the command line the way you'd expect it to
  nvidia_smi = subprocess.Popen(shlex.split("nvidia-smi --query-gpu=timestamp,index,power.draw --format=csv,noheader,nounits -lms 1 -f 'nvidia.log'"))
  # Here's where you would do whatever you want to monitor energy for
  time.sleep(duration)
  # Shut down nvidia_smi
  nvidia_smi.kill()

# Simple function to measure energy
def measure_energy(duration, runtimes=500):
    '''
        GOAL:
        Randomly sample 'runtimes' inputs with normal distribution and
        measure the average energy expenditure

        Input:
            duration (int) time to sleep (filler for real GPU thing to do)
            runtimes (int) number of times to iterate

        Output:
            average joules (float)
            average time (float) # Should be removed in the future, but convenient for now
    '''
    runtime = 0.0
    joules = 0.0
    for i in range(runtimes):
      original_joules = dcpy(joules)
      print("Iteration {0:03d} -- begin".format(i))
      profileStart = time.time()
      energyProfile(duration)
      profileEnd = time.time()
      print("Iteration {0:03d} -- end ({1} seconds)".format(i, profileEnd - profileStart))
      start = time.time()
      # Process nvidia.log for energy data
        # Format: [DATETIME, GPU_ID, WATTS]
      time_sum = 0.0
      with open("nvidia.log", "r") as log:
        # Datetime/timedelta objects parse nvidia_smi timestamps
        time_prev = None
        for x in log.readlines():
          # Sometimes nvidia_smi writes a whole bunch of \x00's
          try:
            entry = [datum.lstrip().rstrip() for datum in x.split(',')]
            # Last entry likely to be incomplete, but all other entries should have length of 3
            if len(entry) != 3:
              break
            # Format: [DATETIME, GPU_ID, WATTS]
              # Select only data from GPU 0 (we may need to make this an argument, but more efficient to tell nvidia-smi via its -i argument)
            if int(entry[1]) != 0:
              continue
            temp_prev = dcpy(time_prev)
            time_prev = datetime.strptime(entry[0], "%Y/%m/%d %H:%M:%S.%f")
            if temp_prev is not None:
              # timedelta can be converted to float by dividing by the desired time resolution
              iteration_time = (time_prev - temp_prev) / timedelta(seconds=1)
              time_sum += iteration_time
              watts = float(entry[2])
              # Joules = Watts * Time(seconds)
              joules += watts * iteration_time
          except Exception:
            # Discard the entry and move on with life
            time_prev = temp_prev
            continue
      finish = time.time()
      runtime += (finish - start)
      print("Iteration {0:03d} -- processed in {1} seconds".format(i, finish-start))
      print("\tIteration {0:03d} -- {1} joules (representing {2} seconds)".format(i, joules - original_joules, time_sum))
      if joules - original_joules < 1.0:
        raise ValueError("No joules detected??")
    return [joules/float(runtimes), runtime/float(runtimes)]

if __name__ == "__main__":
  import sys
  if len(sys.argv) == 1:
    result = measure_energy(3,5)
  elif len(sys.argv) == 2:
    result = measure_energy(int(sys.argv[1]), 5)
  elif len(sys.argv) == 3:
    result = measure_energy(int(sys.argv[1]), int(sys.argv[2]))
  else:
    print("Unsure how to process that commandline")
    print("USAGE: python3 energy.py <sleep_time=3> <runtimes=5>")
    exit()
  print("Average joules = {0}, Average time = {1}".format(result[0], result[1]))

