from skbuild import setup

_cmake_args = [
    "-DNGS_PYTHON_CONFIG=ON",
    "-DCMAKE_BUILD_TYPE=RELEASE",
    "-DENABLE_ELASTICITY=ON",
    "-DENABLE_STOKES=ON",
    "-DENABLE_MIS_AGG=ON",
    "-DENABLE_SPW_AGG=ON",
    "-DENABLE_ROBUST_ELASTICITY_COARSENING=OFF",
    "-DNGS_COMPATIBILITY=ON"
]

setup(
    name="NgsAMG",
    version="0.0.1",
    author="Lukas Kogler",
    license="GNU LESSER GENERAL PUBLIC LICENSE Version3",
    cmake_args=_cmake_args
)