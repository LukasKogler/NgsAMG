
Algebraic Multigrid Preconditioners for NGSolve.

Supports H1 and elasticity problems.

The library can be installed via pip:

    python3 -m pip install git+https://github.com/LukasKogler/NgsAMG.git

You very probably also need to add --no-build-isolation to get the pip install working.

    python -m pip install --no-build-isolation git+https://github.com/LukasKogler/NgsAMG.git

Be aware that this package needs NGSolve to be built with MPI.

The library can also be used with non-python NGSolve builds, that requires building from source using cmake.

For the elasticity AMG component to be fully functional, NGSolve additionally must be built with MAX_SYS_DIM 6 or higher, otherwise direct coarse grid inverse for three dimensional elasticity is not supported.
