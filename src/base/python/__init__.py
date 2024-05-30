import ctypes

ctypes.CDLL("libNgsAMGlib.so", ctypes.RTLD_GLOBAL)

from .NgsAMG import *