/* A Python extension module has four parts:
        + The header file Python.h.
        + The C function we want to expose as the interface from our module.
        + A table mapping the names of our functions as Python developers see
        them to C functions inside the extension module.
        + An initialization function.
*/

/* ------------------------ 1st part: The header fule ----------------------- */
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

/* ------------------------ 2nd part: The C functions ----------------------- */
void vecCompare2Zero(double *input, double *output)
{
    // this function compares a 3d-vector and the 3d zero vector
    for (int i = 0; i < 3; i++)
    {
        if (input[i] > 0)
        {
            output[i] = 1;
        }
        else
        {
            output[i] = 0;
        }
    }
}

void vecAdd(double *input_1, double *input_2, double *output)
{
    // this function computes the summation of 2 3d-vectors
    for (int i = 0; i < 3; i++)
    {
        output[i] = input_1[i] + input_2[i];
    }
}

void vecSub(double *input_1, double *input_2, double *output)
{
    // this function computes the substraction of 2 3d-vectors
    for (int i = 0; i < 3; i++)
    {
        output[i] = input_1[i] - input_2[i];
    }
}

void vecMul(double *input_1, double *input_2, double *output)
{
    // this function computes the product of 2 3d-vectors
    for (int i = 0; i < 3; i++)
    {
        output[i] = input_1[i] * input_2[i];
    }
}

void vecDiv(double *input_1, double *input_2, double *output)
{
    // this function computes the division of 2 3d-vectors
    for (int i = 0; i < 3; i++)
    {
        output[i] = input_1[i] / (input_2[i] + 0.0000001);
    }
}

double func_min(double input1, double input2)
{
    if (input1 <= input2)
    {
        return input1;
    }
    else
    {
        return input2;
    }
}

double func_vecMin(double *input)
{
    // this function returns the minimum entry in a 3d-vector
    double min_value = input[0];

    for (int i = 0; i < 3; i++)
    {
        if (input[i] < min_value)
        {
            min_value = input[i];
        }
    }
    return min_value;
}

int func_vecArgMin(double *input)
{
    // this function returns the position corresponding to the minimum entry of a 3d-vector
    int argmin_value = 0;
    double min_value = func_vecMin(input);
    for (int i = 0; i < 3; i++)
    {
        if (input[i] == min_value)
        {
            argmin_value = i;
        }
    }

    return argmin_value;
}

void get_next_crossing(double *increasing, double *ind_current, double *spacing,
                       double *pt1, double *pt_delta, double *next_crossing)
{
    for (int i = 0; i < 3; i++)
    {
        next_crossing[i] = 0.5 * increasing[i];
    }
    vecAdd(ind_current, next_crossing, next_crossing);
    vecMul(spacing, next_crossing, next_crossing);
    vecSub(next_crossing, pt1, next_crossing);
    vecDiv(next_crossing, pt_delta, next_crossing);
    for (int i = 0; i < 3; i++)
    {
        if (increasing[i] == 0)
        {
            next_crossing[i] = 2;
        }
    }
}

// This part for the filed_line _integral function -----------------------------
double field_line_integral(double *spacing, double ***t_values, double *pt1, double *pt2, int num_pts_y)
{

    double *increasing;
    double *temp_1;
    double *temp_2;
    double *pt_delta;
    double *ind_current;
    double *next_crossing;
    double length = 0;
    double integral = 0;
    double t_next = 0;
    double field_val = 0;
    int ind_coord_next_crossing = 1;
    double t = 0;

    // initializing
    temp_1 = (double *)(calloc(3, sizeof(double)));
    temp_2 = (double *)(calloc(3, sizeof(double)));
    increasing = (double *)(calloc(3, sizeof(double)));
    pt_delta = (double *)(calloc(3, sizeof(double)));
    ind_current = (double *)(calloc(3, sizeof(double)));
    next_crossing = (double *)(calloc(3, sizeof(double)));

    // compute increasing
    vecSub(pt2, pt1, temp_1);
    vecSub(pt1, pt2, temp_2);
    vecCompare2Zero(temp_1, temp_1);
    vecCompare2Zero(temp_2, temp_2);
    vecSub(temp_1, temp_2, increasing);

    // compute length
    for (int i = 0; i < 3; i++)
    {
        length = length + (pt1[i] - pt2[i]) * (pt1[i] - pt2[i]);
    }
    length = sqrt(length);

    // compute pt_delta
    vecSub(pt2, pt1, pt_delta);
    // to avoid dividing by 0
    for (int i = 0; i < 3; i++)
    {
        if (increasing[i] == 0)
        {
            pt_delta[i] = 1;
        }
    }

    // indices of current voxel. order: [x, Ny - y,z]
    vecDiv(pt1, spacing, ind_current);
    for (int i = 0; i < 3; i++)
    {
        ind_current[i] = round(ind_current[i]);
    }

    while (t < 1)
    {

        // get next crossing
        get_next_crossing(increasing, ind_current, spacing, pt1, pt_delta, next_crossing);

        ind_coord_next_crossing = func_vecArgMin(next_crossing);
        t_next = func_min(next_crossing[ind_coord_next_crossing], 1);

        int field_val_x = (int)(ind_current[2]);
        int field_val_y = (int)(num_pts_y - 1 - ind_current[1]);
        int field_val_z = (int)(ind_current[0]);

        field_val = t_values[field_val_x][field_val_y][field_val_z];
        if (field_val != 0.0)
        {
            integral += (t_next - t) * length * field_val;
        }

        // update the current voxel
        t = t_next;
        ind_current[ind_coord_next_crossing] += increasing[ind_coord_next_crossing];
    }

    free((void *)increasing);
    free((void *)temp_1);
    free((void *)temp_2);
    free((void *)pt_delta);
    free((void *)ind_current);
    free((void *)next_crossing);

    return integral;
}
// End of the coding part for field_line_integral in C -------------------------

// Our Python binding to our C function
static PyObject *field_line_integral_C(PyObject *self, PyObject *args)
{

    PyObject *spacing_obj = NULL;
    PyObject *t_values_obj = NULL;
    PyObject *pt1_obj = NULL;
    PyObject *pt2_obj = NULL;

    int num_pts_y;

    if (!PyArg_ParseTuple(args, "OOOOi", &spacing_obj, &t_values_obj, &pt1_obj, &pt2_obj, &num_pts_y))

        return NULL;

    spacing_obj = PyArray_FROM_OTF(spacing_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (spacing_obj == NULL)
    {
        return NULL;
    }
    int spacing_obj_nd = PyArray_NDIM((PyArrayObject *)spacing_obj);
    int spacing_obj_type = PyArray_TYPE((PyArrayObject *)spacing_obj);
    npy_intp *spacing_obj_dims = PyArray_DIMS((PyArrayObject *)spacing_obj);

    t_values_obj = PyArray_FROM_OTF(t_values_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (t_values_obj == NULL)
    {
        return NULL;
    }
    int t_values_nd = PyArray_NDIM((PyArrayObject *)t_values_obj);
    int t_values_type = PyArray_TYPE((PyArrayObject *)t_values_obj);
    npy_intp *t_values_dims = PyArray_DIMS((PyArrayObject *)t_values_obj);

    pt1_obj = PyArray_FROM_OTF(pt1_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (pt1_obj == NULL)
    {
        return NULL;
    }
    int pt1_obj_nd = PyArray_NDIM((PyArrayObject *)pt1_obj);
    int pt1_obj_type = PyArray_TYPE((PyArrayObject *)pt1_obj);
    npy_intp *pt1_obj_dims = PyArray_DIMS((PyArrayObject *)pt1_obj);

    pt2_obj = PyArray_FROM_OTF(pt2_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (pt2_obj == NULL)
    {
        return NULL;
    }

    double *spacing;
    double ***t_values;
    double *pt1;
    double *pt2;

    int check_spacing = PyArray_AsCArray(&spacing_obj, (double *)&spacing, spacing_obj_dims, spacing_obj_nd, PyArray_DescrFromType(spacing_obj_type));
    int check_t_values = PyArray_AsCArray(&t_values_obj, (double ***)&t_values, t_values_dims, t_values_nd, PyArray_DescrFromType(t_values_type));
    int check_pt1 = PyArray_AsCArray(&pt1_obj, (double *)&pt1, pt1_obj_dims, pt1_obj_nd, PyArray_DescrFromType(pt1_obj_type));
    int check_pt2 = PyArray_AsCArray(&pt2_obj, (double *)&pt2, pt1_obj_dims, pt1_obj_nd, PyArray_DescrFromType(pt1_obj_type));

    if (check_spacing < 0 || check_t_values < 0 || check_pt1 < 0 || check_pt2 < 0)
    {
        PyErr_SetString(PyExc_TypeError, "error converting to c array");

        return NULL;
    }

    // return integral
    double integral = field_line_integral(spacing, t_values, pt1, pt2, num_pts_y);

    Py_DECREF(spacing_obj);
    Py_DECREF(t_values_obj);
    Py_DECREF(pt1_obj);
    Py_DECREF(pt2_obj);

    PyArray_Free(spacing_obj, spacing);
    PyArray_Free(t_values_obj, (double **)t_values);
    PyArray_Free(pt1_obj, pt1);
    PyArray_Free(pt2_obj, pt2);

    return Py_BuildValue("d", integral);
}

/* ------------------ 3rd part: The method mapping table ------------------ */
static PyMethodDef myMethods[] = {
    {"field_line_integral_C", field_line_integral_C, METH_VARARGS, "Sum of matrices"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef grid_utilities = {
    PyModuleDef_HEAD_INIT,
    "field line integral",
    "Test Module",
    -1,
    myMethods};

/* ------------------ 4nd part: The method mapping table ------------------ */
PyMODINIT_FUNC PyInit_grid_utilities(void)
{
    import_array();
    return PyModule_Create(&grid_utilities);
}