#include <Python.h>

static PyObject *method_myadd(PyObject *self, PyObject *args){
    int x, y, z = -1;

    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "ii", &x, &y)){
            return NULL;
    }

    /* The actual bit of code I need */
    z = x + y;

    return PyLong_FromLong(z);
}

static PyMethodDef myaddMethods[] = {
    {"myadd", method_myadd, METH_VARARGS, "Python interface for myadd C library function"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef myaddmodule = {
    PyModuleDef_HEAD_INIT,
    "myadd",
    "Python interface for the myadd C library function",
    -1,
    myaddMethods
};

PyMODINIT_FUNC PyInit_myadd(void) {
    return PyModule_Create(&myaddmodule);
}
