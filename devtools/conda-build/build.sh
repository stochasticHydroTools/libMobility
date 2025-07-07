set -euxo pipefail
rm -rf build || true
TMPDIR=$(mktemp -d)
SRC_DIR=$(pwd)
cd $TMPDIR

NANOBIND_DIR=$(${PYTHON} -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nanobind/cmake
CMAKE_FLAGS="${CMAKE_ARGS} -DCMAKE_PREFIX_PATH=${PREFIX} -DCMAKE_INSTALL_PREFIX=${PREFIX} -DCMAKE_BUILD_TYPE=Release"
CMAKE_FLAGS+=" -DPython_EXECUTABLE=${PYTHON} -DCMAKE_VERBOSE_MAKEFILE=ON"
CMAKE_FLAGS+=" -Dnanobind_DIR=${NANOBIND_DIR}"
cmake $SRC_DIR ${CMAKE_FLAGS}
make install -j$CPU_COUNT
rm -rf $TMPDIR
