#!/bin/bash
# Set OpenBLAS thread limits to prevent memory allocation errors
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run the webservice
python misc/webservice.py "$@"
