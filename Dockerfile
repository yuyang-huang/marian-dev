FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# install Intel MKL for faster matrix operations
RUN apt-key adv --fetch-keys https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB \
  && echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list

COPY . /opt/marian-dev
WORKDIR /opt/marian-dev

RUN BUILD_DEPS=" \
    build-essential \
    ca-certificates \
    doxygen \
    git \
    intel-mkl-2020.0-088 \
    libboost-all-dev \
    libgoogle-perftools-dev \
    libprotobuf-dev \
    libssl-dev \
    protobuf-compiler \
    python3-pip \
  " \
  && apt-get update && apt-get install -y --no-install-recommends $BUILD_DEPS \
  # install newer version of CMake: marian depends on CMake>=3.12.2
  && pip3 install 'cmake==3.13.3' \
  # build Marian
  && mkdir build && cd build \
  && cmake \
    -DUSE_FBGEMM=on \
    -DUSE_STATIC_LIBS=on \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_SENTENCEPIECE=on \
    -DCOMPILE_SERVER=on \
    .. \
  && make -j4 \
  && mv marian-server /usr/local/bin/marian-server-avx512 \
  && mv marian* /usr/local/bin \
  && cd .. && rm -rf build \
  # build AVX2 server
  && mkdir build && cd build \
  && cmake \
    -DUSE_FBGEMM=on \
    -DUSE_STATIC_LIBS=on \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_SENTENCEPIECE=on \
    -DCOMPILE_SERVER=on \
    -DBUILD_ARCH=broadwell \
    .. \
  && make -j4 marian_server \
  && mv marian-server /usr/local/bin/marian-server-avx2 \
  && cd .. && rm -rf build \
  # clean up
  && apt-get remove -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false $BUILD_DEPS \
  && rm -rf /var/lib/apt/lists/*
