set -euxo pipefail
rm -rf build || true
TMPDIR=$(mktemp -d)
SRC_DIR=$(pwd)
cd $TMPDIR

CMAKE_FLAGS="${CMAKE_ARGS} -DCMAKE_PREFIX_PATH=${PREFIX} -DCMAKE_INSTALL_PREFIX=${PREFIX} -DCMAKE_BUILD_TYPE=Release"
CMAKE_FLAGS+=" -DPython_EXECUTABLE=${PYTHON}"
cmake $SRC_DIR ${CMAKE_FLAGS}
make install -j$CPU_COUNT
rm -rf $TMPDIR
