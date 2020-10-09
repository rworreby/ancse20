# Requirements
  * A C++ compiler (GCC, Clang and MSVC tested).
  * Either MATLAB or Python (with numpy + matplotlib) for plotting.


# Building with CMake

We strongly recommend building the source using cmake.

## Building with cmake from the command line

In the template_code folder, first make a new build folder:

    mkdir build
    cd build

then issue cmake:

    cmake ..

You can build the whole program using

    make

## Running the executables

If you used cmake, you can now run the executables using

    ./linear_transport

## About Eigen

We have bundled the template and solution code with the Eigen library (see
http://eigen.tuxfamily.org/). For license information on using Eigen, see the
contents of the folder LicenseEigen. *All files under the folder Eigen are
subject to the licenses used by Eigen*.

