TMPDIR=$(mktemp -d)
SRC_DIR=$(pwd)
cd $TMPDIR
cmake $SRC_DIR -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release
make install -j$CPU_COUNT
rm -rf $TMPDIR
