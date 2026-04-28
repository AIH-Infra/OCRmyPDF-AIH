@echo off
REM Set OpenBLAS thread limits to prevent memory allocation errors
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1

REM Run the webservice
python misc\webservice.py %*
