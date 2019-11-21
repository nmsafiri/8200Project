import time, threading, subprocess, shlex, numpy as np
from datetime import datetime, timedelta
from copy import deepcopy as dcpy

# Creates an nvidia-smi process that logs our energy metrics (can be killed at-will)
class energyProfile(threading.Thread):
  def __init__(self, *args, **kwargs):
    super(energyProfile, self).__init__(*args, **kwargs)
    self._stopThread = threading.Event()
  def stopThread(self): # Function you call to stop the thread
    self._stopThread.set()
  def run(self): # Thread's job to do when running
    # Create log of output
    proc = subprocess.Popen(shlex.split("nvidia-smi --query-gpu=timestamp,index,power.draw --format=csv,noheader,nounits -lms 1 -f 'nvidia.log'"))
    while True:
      if self._stopThread.isSet():
        proc.kill()
        return
      time.sleep(1)

def splitJoinTime():
  nvidiaEnergy = energyProfile()
  nvidiaEnergy.start()
  start = time.time()
  time.sleep(3)
  finish = time.time()
  nvidiaEnergy.stopThread()
  nvidiaEnergy.join()
  return [start, finish]

def measure_energy(runtimes=500):
    '''
        Measure energy of 'thing_to_do'
        Randomly sample 'runtimes' inputs with normal distribution and
        measure the energy

        Input:
            `thing_to_do`: thing to be measured
            `input_for_thing`: (list) input

        Output:
            average joules (float)
    '''
    runtime = 0.0
    joules = 0.0
    for i in range(runtimes):
      print("Iteration {0:03d} -- begin".format(i))
      start, finish = splitJoinTime()
      print("Iteration {0:03d} -- end ({1} seconds)".format(i, finish-start))
      runtime += (finish - start)
      nvidia_smi_data = np.genfromtxt("nvidia.log", delimiter=",", dtype=str,\
                                      autostrip=True, invalid_raise=False,\
                                      unpack=False,\
                                      converters={\
                                        0: lambda x: str(x),\
                                        1: lambda y: int(y),\
                                        2: lambda z: float(z)})
      time_prev = None
      iteration_time = timedelta()
      iteration_watts = 0.0
      ticks = 0
      for line in nvidia_smi_data:
        if line[1] != 0:
          continue
        if time_prev is not None:
          temp_prev = dcpy(time_prev)
          time_prev = datetime.strptime(line[0][2:-1], "%Y/%m/%d %H:%M:%S.%f")
          iteration_time += time_prev - temp_prev
        else:
          time_prev = datetime.strptime(line[0][2:-1], "%Y/%m/%d %H:%M:%S.%f")
        iteration_watts += line[2]
        ticks += 1
      joules += (iteration_watts * (iteration_time / timedelta(microseconds=1)) / 1e6) / ticks
    return [joules/float(runtimes), runtime/float(runtimes)]

if __name__ == "__main__":
  result = measure_energy(5)
  print("Average joules = {0}, Average time = {1}".format(result[0], result[1]))

