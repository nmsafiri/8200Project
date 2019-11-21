import time, threading, subprocess, shlex, numpy as np
from datetime import datetime, timedelta
from copy import deepcopy as dcpy

# Creates an nvidia-smi process that logs our energy metrics (can be killed at-will)
class energyProfile(threading.Thread):
  def __init__(self, duration, *args, **kwargs):
    super(energyProfile, self).__init__(*args, **kwargs)
    self._stopThread = threading.Event()
    self.duration = duration
  def run(self): # Thread's job to do when running
    # Create log of output using nvidia_smi
      # shlex.split() helps Popen() read the command line the way you'd expect it to
    nvidia_smi = subprocess.Popen(shlex.split("nvidia-smi --query-gpu=timestamp,index,power.draw --format=csv,noheader,nounits -lms 1 -f 'nvidia.log'"))
    # Here's where you would do whatever you want to monitor energy for
    time.sleep(self.duration)
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
      print("Iteration {0:03d} -- begin".format(i))
      start = time.time()
      nvidiaEnergy = energyProfile(duration)
      nvidiaEnergy.start()
      nvidiaEnergy.join()
      finish = time.time()
      print("Iteration {0:03d} -- end ({1} seconds)".format(i, finish-start))
      runtime += (finish - start)
      # Numpy function reads the csv and returns numpy array of the data
        # genfromtxt is tolerant to incomplete records, which often occur since we likely kill nvidia_smi mid-write
      nvidia_smi_data = np.genfromtxt("nvidia.log", delimiter=",", dtype=str,\
                                      autostrip=True, invalid_raise=False,\
                                      unpack=False,\
                                      converters={\
                                        0: lambda x: str(x),\
                                        1: lambda y: int(y),\
                                        2: lambda z: float(z)})
      # Use datetime and timedelta objects to parse the nvidia_smi timestamps correctly
      time_prev = None
      for line in nvidia_smi_data:
        # LINE FORMAT: [DATETIME, GPU_ID, WATTS]
        # Select only data from GPU 0 (we may need to make this an argument, but more efficient to tell nvidia-smi via its -i argument)
        if line[1] != 0:
          continue
        if time_prev is not None:
          temp_prev = dcpy(time_prev)
          time_prev = datetime.strptime(line[0][2:-1], "%Y/%m/%d %H:%M:%S.%f")
          iteration_time = time_prev - temp_prev
          watts = line[2]
          # Joules = Watts * Time(seconds)
          # timedelta can be converted to float by dividing by the desired time resolution
          joules += watts * (iteration_time / timedelta(seconds=1))
        else:
          time_prev = datetime.strptime(line[0][2:-1], "%Y/%m/%d %H:%M:%S.%f")
    return [joules/float(runtimes), runtime/float(runtimes)]

if __name__ == "__main__":
  result = measure_energy(3,5)
  print("Average joules = {0}, Average time = {1}".format(result[0], result[1]))

