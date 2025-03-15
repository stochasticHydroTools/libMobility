from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import subprocess
import os
import sys
import multiprocessing

try:
    version = (
        subprocess.check_output(["git", "describe", "--abbrev=0", "--tags"])
        .strip()
        .decode("utf-8")
    )
except:
    print("Failed to retrieve the current version, defaulting to 0")
    version = "0"


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        install_dir = os.path.abspath(
            sys.prefix
        )  # Use sys.prefix to install in the environment's prefix

        double_precision = os.environ.get("DOUBLEPRECISION", "0")  # Default to "0" if not set
        if double_precision not in {"0", "1"}:
            raise ValueError("Environment variable DOUBLEPRECISION must be '0' or '1'. It is 0 by default- set to 1 to compile in double precision.")

        cmake_args = [
            "-DCMAKE_INSTALL_PREFIX=" + install_dir,  # Point to the install directory
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DCMAKE_BUILD_TYPE=" + ("Debug" if self.debug else "Release"),
            f"-DDOUBLEPRECISION={double_precision}",
        ]

        num_jobs = os.cpu_count()
        build_args = ["--config", "Release", f"-j{num_jobs}"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "install"] + build_args,
            cwd=self.build_temp,
        )


setup(
    name="libMobility",
    version=version,
    packages=find_packages(),
    ext_modules=[CMakeExtension("libMobility")],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
