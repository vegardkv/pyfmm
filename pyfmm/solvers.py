# MIT License
#
# Copyright (c) 2016 Vegard Kvernelv
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np


def _expand_nesw(arr):
    out_arr = arr.copy()
    out_arr[:-1, :] |= arr[1:]
    out_arr[1:, :] |= arr[:-1]
    out_arr[:, :-1] |= arr[:, 1:]
    out_arr[:, 1:] |= arr[:, :-1]
    return out_arr


def _approximate_distance(current_values, fvals, certain_values=None, to_consider=None):
    """
    :param current_values:
    :param fvals:
    :param certain_values:
    :param to_consider: implicit requirement: the frame of "to_consider" must contain only zeroes.
    :return:
    """
    nx, ny = current_values.shape
    if certain_values is None:
        certain_values = np.ones((nx, ny), dtype=np.bool)
        certain_values[np.isinf(current_values)] = False
    if to_consider is None:
        to_consider = (certain_values == False)
        to_consider[0, :] = False
        to_consider[-1, :] = False
        to_consider[:, 0] = False
        to_consider[:, -1] = False
    current_values[np.invert(certain_values)] = np.inf

    cons_xl = np.concatenate((to_consider[1:,:], np.zeros((1, ny), dtype=np.bool)), axis=0)
    cons_xr = np.concatenate((np.zeros((1, ny), dtype=np.bool), to_consider[:-1, :]), axis=0)
    u_xl = current_values[cons_xl]
    u_xr = current_values[cons_xr]
    uh = np.minimum(u_xl, u_xr)

    cons_yl = np.concatenate((to_consider[:, 1:], np.zeros((nx, 1), dtype=np.bool)), axis=1)
    cons_yr = np.concatenate((np.zeros((nx, 1), dtype=np.bool), to_consider[:, :-1]), axis=1)
    u_yl = current_values[cons_yl]
    u_yr = current_values[cons_yr]
    uv = np.minimum(u_yl, u_yr)

    uv_uh = uv + uh
    aprx = (uv_uh / 2 + 0.5 * np.sqrt(
        np.square(uv_uh) -
        2 * (np.square(uv) + np.square(uh) - 1.0 / fvals[to_consider])))

    invalid_aprx = np.isinf(aprx) + np.isnan(aprx)
    aprx[invalid_aprx] = np.minimum(uh[invalid_aprx], uv[invalid_aprx]) + 1.0/fvals[to_consider][invalid_aprx]
    current_values[to_consider] = np.minimum(aprx, current_values[to_consider])
    return current_values


def march(mask, distance=None, speed=None, batch_size=1):
    """
    Compute the distance from a boundary using a variant of the Fast Marching Method (FMM). The regular FMM accepts one
    point value at each iteration, however for certain problems, a speed up can be achieved by accepting the n smallest
    values at each iteration instead of just the smallest. This may, in some cases, be inaccurate, but the speed-up can
    be significant.
    :param mask: A boolean array describing the boundary (or known distances values if 'distance' is given). If
     mask[i,j] is True, then the distance to the boundary from grid point (i, j) is assumed known, and given by
     distance[i,j] (if distance is not given, it is set to 0). All False values are to be computed.
    :param distance: Prior distances to the boundary. If corresponding 'mask' element is True, then the value remains
     unchanged.
    :param speed: Speed at which grid node [i, j] can be traversed.
    :param batch_size: Maximum number of values to accept at each iteration. The value 1 yields the regular FMM, and
     using np.inf means accepting as much as possible at each step (although this can produce some unfortunate edge
     effects.
    :return: A tuple where the first element is the computed distances, and the second is a boolean array with True for
     all grid points that have been computed by the method (this can be useful if parts of the grid have speed 0 and is
     thus inaccessible).
    """
    mesh_shape = mask.shape
    if speed is None:
        speed = np.ones(mesh_shape, dtype=np.float)
    if distance is None:
        uu = np.ones(mesh_shape) * np.inf
        uu[mask] = 0
        certain = mask.copy()
    else:
        uu = distance.copy()
        certain = mask.copy()
        uu[np.invert(certain)] = np.inf

    uu_padded = np.lib.pad(uu, (1, 1), mode='constant', constant_values=np.inf)
    certain_padded = np.lib.pad(certain, (1, 1), mode='constant', constant_values=False)
    speed_padded = np.lib.pad(speed, (1, 1), mode='constant', constant_values=0.0)
    mesh_shape = uu_padded.shape

    # Stop if no progress is made
    current_count_certain = np.count_nonzero(certain_padded)
    previous_count_certain = current_count_certain + 1

    # Kernel
    while current_count_certain - previous_count_certain != 0:
        consider_next = np.logical_xor(_expand_nesw(certain_padded), certain_padded)
        consider_next[0 , :] = False
        consider_next[-1, :] = False
        consider_next[: , 0] = False
        consider_next[: ,-1] = False
        uu_padded = _approximate_distance(uu_padded, speed_padded, certain_padded, to_consider=consider_next)
        if batch_size == 1:  # guaranteed
            if np.any(consider_next):
                p = np.argmin(uu_padded[consider_next])
                unr = np.argwhere(consider_next.flatten())[p]
                unraveled_p = np.unravel_index(unr, mesh_shape)
                certain_padded[unraveled_p] = True
        elif batch_size == np.inf:  # greedy
            certain_padded[np.invert(np.isinf(uu_padded))] = True
        else:  # heuristic
            ps = np.argpartition(uu_padded[consider_next], min(batch_size, np.count_nonzero(consider_next)) - 1)
            unr = np.argwhere(consider_next.flatten())[ps[:(batch_size - 1)]]
            unraveled_ps = np.unravel_index(unr, mesh_shape)
            certain_padded[unraveled_ps] = np.invert((np.isinf(uu_padded[unraveled_ps]) + np.isnan(uu_padded[unraveled_ps])))
        previous_count_certain = current_count_certain
        current_count_certain = np.count_nonzero(certain_padded)
    return uu_padded[1:-1,1:-1], certain_padded[1:-1,1:-1]
