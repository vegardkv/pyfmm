# pyfmm
Python module implementing the Fast Marching Method. Only dependency is numpy 1.8+.

The implementation uses mostly boolean arrays for accessing and updating values. Instead of accepting only the smallest value at each iteration (step 3, https://en.wikipedia.org/wiki/Fast_marching_method), one may accept an arbitrary number of values at each step. This can speed up the computations considerably, but may in some cases be inaccurate (especially if the speed varies alot).

This is work under development.
