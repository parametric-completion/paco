#!/bin/bash

# install polyfit (depricated for wheel in requirements.txt)
apt install libgmp3-dev libcgal-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
echo "Installing polyfit..."
cd utils && git clone https://github.com/LiangliangNan/PolyFit.git polyfit && cd polyfit
mkdir release && cd release
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j8
