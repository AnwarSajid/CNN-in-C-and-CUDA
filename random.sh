RSEED1=`od -An -N2 -i /dev/random`
make clean
make
./conv_gpu.out $RSEED1
