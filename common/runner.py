"""
Changelog:

2021/08/07: When the invoked function returns a row vector, it is left as a row
vector, not converted to a column. See x22Dpythonarray

"""


import os
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
from IPython.core.debugger import set_trace
import traceback
import re
from collections import OrderedDict
import importlib

output_folder = "grading_reports"

mat_eng = None  # MATLAB engine
matlab_module = None

class Runner():
    """Objects of this class allow the user to execute functions in a
    file, which can be written either in Python or MATLAB."""

    available_langs = {"m", "py"}

    def load_matlab(self):

        global mat_eng
        if mat_eng is None:  # load only if not already loaded
            print("Loading MATLAB...", end='', flush=True)
            # Start MATLAB and add the aforementioned folder to the path
            mat_eng = matlab_module.engine.start_matlab()
            mat_eng.addpath(self.input_folder)

            # Test that it works
            _ = mat_eng.randn(4, 2)
            print(" done!")
        return mat_eng

    def __init__(self,                 
                 input_folder, # relative path
                 filename,
                 py_class_name="Functions",
                 ):
        global mat_eng
        global matlab_module

        assert filename.count(".") == 1
        name, ext = filename.split(".")
        if ext not in self.available_langs:
            raise ValueError(
                "Only filenames with extension .m or .py are supported")
        self.lang = ext
        self.input_folder = input_folder
        if self.lang == "m":            
            import matlab.engine
            matlab_module = matlab

            mat_eng = self.load_matlab()
            class_name = name
            self.backend_class = getattr(mat_eng, class_name)
            
        elif self.lang == "py":
            if input_folder != "":
                input_folder += "."
            module = importlib.import_module(input_folder + name)
            self.backend_class = getattr(module, py_class_name)
            
    def run(self, function_name, od_data, nargout=1):
        """Invokes the function with name `function_name` with input arguments
           as contained in the ordered dict `od_data`. It returns a tuple
           with whatever the function returns converted to 2D np.arrays.
        """

        assert isinstance(nargout, int)
        assert nargout >= 1

        if self.lang == "m":
            data = [self.pythonarray2mldouble(val) for val in od_data.values()]
            fun = getattr(self.backend_class, function_name)
            out = fun(*data, nargout=nargout)
        elif self.lang == "py":
            out = getattr(self.backend_class, function_name)(**od_data)
            #out = self.x22Dpythonarray(out)

        if out is None:
            raise ValueError(
                f"Function {function_name} in {self.backend_class} returned None"
            )

        # Convert to tuple
        if isinstance(out, tuple):
            return tuple(self.x22Dpythonarray(out_el) for out_el in out)
        else:
            return (self.x22Dpythonarray(out), )

    @classmethod
    def pythonarray2mldouble(cls, python_array):

        np_array = cls.x22Dpythonarray(python_array)

        return matlab_module.double(np_array.tolist())

    @classmethod
    def x22Dpythonarray(cls, x):
        # Converts "whatever" to a 2D python array. If a vector, then it
        # becomes a column vector.

        if type(x) == np.ndarray:
            np_array = x
        elif type(x) == matlab_module.double:
            np_array = cls.mldouble2nparray(x)
        else:
            np_array = np.array(x)

        # Expand dimensions if necessary
        if len(np_array.shape) == 0:
            np_array = np.expand_dims(np.expand_dims(np_array, 0), 1)
        if len(np_array.shape) == 1:
            np_array = np.expand_dims(np_array, 1)  # column vector by default
        if len(np_array.shape) > 2:
            raise Exception("Not implemented")

        #if np_array.shape[0] == 1 and np_array.shape[1] > 1:
        #    np_array = np_array.T

        return np_array

    @staticmethod
    def mldouble2nparray(mldouble):

        return np.asarray(mldouble)
        # old
        # if not ('_data' in dir(mldouble)):
        #     raise AssertionError('Function not implemented for complex data')

        # nparray = np.array(mldouble._data)

        # if len(mldouble._size) == 2:
        #     return np.reshape(nparray, mldouble._size)  #, (2, 1))
        # else:
        #     raise Exception(
        #         'Function implemented for vectors and matrices only')
