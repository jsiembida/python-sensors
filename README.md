# python-sensors

Simple, pure python utility for linux sensors.

Originally, I needed an easy to use utility to monitor CPU load, freq and power consumption in linux.
So I hacked a python script to do just that. Python-sensors is a cleaned up version thereof.

- It is a single python file.
- No extra libraries needed, it only requires Python 3.3+.
- No extra utilities needed, it extracts all the info it needs directly from `procfs` and `sysfs`.
- Therefore, it handles sensors that are loaded and exposed via sysfs.
- Tested with Ubuntu and Raspberry Pi OS.
- Even though `sysfs` is cheap to access, effort was made to further minimize
syscalls and I/O so that overall overhead should be negligible.
- Can be used as a command line utility as well as a python library.


## Quick start

Clone this repo or just copy `sense.py` from it, because that's all there is to it.


## Command line

Example use on Raspberry Pi:

```shell
root@raspberrypi:~# ./sense.py --output-format=table --count 1

2022-06-01T21:51:49.610166+01:00

      resource   instance       freq      temp    usage   description
                                                                     
           cpu          1   1.400GHz             0.000%              
           cpu          2   1.400GHz             0.000%              
           cpu          3   1.400GHz             0.000%              
           cpu          4   1.400GHz             0.000%              
   cpu_thermal        0/1              39.704C                       
```

Use `--help` option to find out more.


## API

The intention is to use the exposed generator that returns a list of datapoints:

```python
>>> from sense import Sensors
>>> sensors = iter(Sensors().read(power_caps=False, hardware_sensors=False))
>>> datapoints = next(sensors)
>>> print(datapoints[0])
{'resource': 'cpu', 'instance': (1,), 'description': None, 'quantity': 'freq', 'value': 900164000, 'unit': 'Hz'}
```

Please see `sense.py` for a nicer pattern and more API details.
Note, datapoints returned by this API endpoint always use base units like `Hz`, `K`
and `W` even though linux `sysfs` exposes these quantities in a different fashion.
