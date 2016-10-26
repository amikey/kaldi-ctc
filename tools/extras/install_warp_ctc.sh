#!/bin/bash
# install warp-ctc

! which cmake >/dev/null  && \
   echo "cmake is not installed, this will not work.  Ask your sysadmin to install it" && exit 1;

if [ ! -d warp-ctc ];then
	git clone https://github.com/lifeiteng/warp-ctc.git || exit 1
fi

cd warp-ctc; git pull
mkdir -p build; cd build; cmake ../; make -j1 || exit 1
./test_cpu || exit 1;
./test_gpu || exit 1;
cd ../../
