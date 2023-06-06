import numpy as np
#import tensorflow as tf
from collections import OrderedDict
import pathlib

# project folder
fo_project = pathlib.Path(__file__).parent.resolve().parent.resolve()


def is_vector(obj):

    array = np.array(obj)
    return array.ndim == 1


def mat_argmax(m_A):
    """ returns tuple with indices of max entry of matrix m_A"""

    num_cols = m_A.shape[1]
    assert m_A.ndim == 2

    ind = np.argmax(m_A)

    row = ind // num_cols
    col = ind % num_cols
    return (row, col)


def mat_argmin(m_A):
    """ returns tuple with indices of min entry of matrix m_A"""

    num_cols = m_A.shape[1]
    assert m_A.ndim == 2

    ind = np.argmin(m_A)

    row = ind // num_cols
    col = ind % num_cols
    return (row, col)


def print_time(start_time, end_time):
    td = end_time - start_time
    hours = td.seconds // 3600
    reminder = td.seconds % 3600
    minutes = reminder // 60
    seconds = (td.seconds - hours * 3600 -
               minutes * 60) + td.microseconds / 1e6
    time_str = ""
    if td.days:
        time_str = "%d days, " % td.days
    if hours:
        time_str = time_str + "%d hours, " % hours
    if minutes:
        time_str = time_str + "%d minutes, " % minutes
    if time_str:
        time_str = time_str + "and "

    time_str = time_str + "%.3f seconds" % seconds
    #set_trace()
    print("Elapsed time = ", time_str)


def empty_array(shape):
    return np.full(shape, fill_value=None, dtype=float)


def project_to_interval(x, a, b):
    assert a <= b
    return np.max([np.min([x, b]), a])


def watt_to_dbW(array):
    array = np.array(array)
    assert (array > 0).all()

    return 10 * np.log10(array)


def watt_to_dbm(array):
    return watt_to_dbW(array) + 30


def dbm_to_watt(array):
    return 10**((array - 30) / 10)


def natural_to_dB(array):  # array is power gain
    return 10 * np.log10(array)


def dB_to_natural(array):  # array is power gain
    return 10**(np.array(array) / 10)


def sum_db(array, *args, **kwargs):
    """wrapper for np.sum where `array` is converted to natural before summing
    and back to dB after summing."""
    return natural_to_dB(np.sum(dB_to_natural(array), *args, **kwargs))


# def save_l_var_vals(l_vars):
#     # returns a list of tensors with the values of the
#     # variables in the list l_vars.
#     l_vals = []
#     for var in l_vars:
#         l_vals.append(tf.convert_to_tensor(var))
#     return l_vals


def restore_l_var_vals(l_vars, l_vals):

    assert len(l_vars) == len(l_vals)
    # assigns the value l_vals[i] to l_vars[i]
    for var, val in zip(l_vars, l_vals):
        var.assign(val)


class FifoUniqueQueue():
    """FIFO Queue that does not push a new element if it is already in the queue. Pushing an element already in the queue does not change the order of the queue.
    It seems possible to implement this alternatively as a simple list.
"""

    def __init__(self):
        self._dict = OrderedDict()

    def put(self, key):
        self._dict[key] = 0  # dummy value, for future usage

    def get(self):
        # Returns oldest item
        return self._dict.popitem(last=False)[0]

    def empty(self):
        return len(self._dict) == 0


def nearest_row(m_matrix, v_row, err_tol=.1):
    """Returns the index of the row of m_matrix that has the smallest Euclidean distance to v_row."""

    ind = np.argmin(np.sum((m_matrix - v_row)**2, axis=1))

    err = np.linalg.norm(m_matrix[ind] - v_row)
    if err > err_tol:
        return None

    return ind


def get_components(m_ajacency):
    """ 
    Args:

    `m_adjacency`: num_nodes x num_nodes. `m_adjacency[m,n]` is True if it is
    possible to go from node m to node n. 

    Returns:

    `ll_components`: list with num_components entries. The n-th entry contains
    the indices of the nodes in the n-th component. 
    
    """

    def get_nodes_in_same_component(ind_node):
        """Returns a list with the indices of the nodes in the same component as
        the node with index `ind_node`"""

        def get_new_descendants(ind_node, current_descendants):
            """Returns `current_descendants` union the descendants of `ind_node`"""

            s_children = set(np.where(m_ajacency[ind_node])[0])
            s_new_descendants = s_children - current_descendants
            s_descendants = current_descendants.union(s_new_descendants)

            for new_descendant in s_new_descendants:
                s_descendants = s_descendants.union(
                    get_new_descendants(new_descendant, s_descendants))

            return s_descendants

        return get_new_descendants(ind_node, set([ind_node]))

    if np.any(m_ajacency != m_ajacency.T):
        """In this case, we need to find also the ascendants; check with [[0, 1,
        0, 0], [1, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0]]"""
        raise NotImplementedError

    s_nodes_remaining = set(range(len(m_ajacency)))
    ll_components = []
    while len(s_nodes_remaining):
        ind_node_now = s_nodes_remaining.pop()
        s_new = get_nodes_in_same_component(ind_node_now)
        ll_components.append(list(s_new))
        s_nodes_remaining = s_nodes_remaining - s_new

    return ll_components


def find_component(ind_node, ll_components):
    """
        Find the index of the component containing ind_node

    """

    for ind in range(len(ll_components)):
        if ind_node in ll_components[ind]:
            return ind
    raise ValueError(f"Node {ind_node} is not in any component")


# print('hello')
# m_A = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
# m_A = m_A + m_A.T
# ll_components = get_components(m_A != 0)

# for l_component in ll_components:
#     print(l_component)