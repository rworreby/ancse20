cd build
rm -r testing-*
make -j
./fvm_scalar
cd ../
python3 plot.py build/testing-
