from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import pybind11
import setuptools
import argparse

# [utils]
class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the `get_include()`
    method can be invoked. """

    def __str__(self):
        return pybind11.get_include()

class CustomBuildExt(build_ext):
    def build_extension(self, ext):
        # Change the output directory to the current directory
        ext_path = self.get_ext_fullpath(ext.name)
        ext_path = os.path.join(os.getcwd(), os.path.basename(ext_path))
        self.build_lib = os.path.dirname(ext_path)
        super().build_extension(ext)

def cpp_extension(module_name, source_files, include_dirs=None, extra_compile_args=None):
    if include_dirs is None:
        include_dirs = []
    if extra_compile_args is None:
        extra_compile_args = []

    include_dirs.append(get_pybind_include())

    return Extension(
        module_name,
        source_files,
        include_dirs=include_dirs,
        language='c++',
        extra_compile_args=extra_compile_args,
    )

if __name__ == '__main__':
    # [db]
    available_extensions = {
        'itp_state': cpp_extension(
            'fast_math.itp_state.itp_state',
            ['fast_math/itp_state/itp_state.cpp'],
            extra_compile_args=['-std=c++17']
        ),
        'ring_buffer': cpp_extension(
            'fast_math.container.ring_buffer',
            ['fast_math/container/ring_buffer.cpp'],
            extra_compile_args=['-std=c++17']
        ),
    }

    # [logic]
    parser = argparse.ArgumentParser()
    parser.add_argument('--ex', nargs='*', help='List of extensions to build')
    args, unknown = parser.parse_known_args()

    selected_extensions = []
    if hasattr(args, "ex") and args.ex is not None:
        for ext in args.ex:
            print(f'ext: {ext}')
            if ext in available_extensions:
                selected_extensions.append(available_extensions[ext])
    print(f'seleted_extensions: {selected_extensions}')

    sys.argv = [sys.argv[0]] + unknown
    # [setups]

    setup(
        name='fast_math',
        version='0.1',
        packages=setuptools.find_packages(),
        extras_require={},
        ext_modules=selected_extensions,
        cmdclass={'build_ext': CustomBuildExt},
    )
