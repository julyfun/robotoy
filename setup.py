from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import setuptools
import argparse

# [utils]
class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the `get_include()`
    method can be invoked. """

    def __str__(self):
        import pybind11
        return pybind11.get_include()

class CustomBuildExt(build_ext):
    def build_extension(self, ext):
        # Change the output directory to the current directory
        ext_path = self.get_ext_fullpath(ext.name)
        ext_path = os.path.join(os.getcwd(), os.path.basename(ext_path))
        self.build_lib = os.path.dirname(ext_path)
        super().build_extension(ext)

    def copy_extensions_to_source(self):
        # This is crucial for sdist to include the .so files
        build_dir = os.path.abspath(self.build_lib)
        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            src = os.path.join(build_dir, filename)
            dest = os.path.join(os.path.relpath(self.build_lib, '.'), filename) # Correct destination

            # Ensure the destination directory exists
            dest_dir = os.path.dirname(dest)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            self.copy_file(src, dest)

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
            'robotoy.itp_state.itp_state',
            ['robotoy/itp_state/itp_state.cpp'],
            extra_compile_args=['-std=c++17']
        ),
        'ring_buffer': cpp_extension(
            'robotoy.container.ring_buffer',
            ['robotoy/container/ring_buffer.cpp'],
            extra_compile_args=['-std=c++17']
        ),
    }

    # [logic]
    parser = argparse.ArgumentParser()
    parser.add_argument('--ex', nargs='*', help='List of extensions to build')
    args, unknown = parser.parse_known_args()

    selected_extensions = []
    if hasattr(args, "ex") and args.ex is not None:
        if 'all' in args.ex:
            selected_extensions = list(available_extensions.values())
        else:
            for ext in available_extensions.keys():
                if ext in args.ex:
                    selected_extensions.append(available_extensions[ext])
    print(f'seleted_extensions: {selected_extensions}')

    # [c++]
    install_requires = []
    setup_requires = []
    if any(ext.language == 'c++' for ext in selected_extensions):
        setup_requires.append('pybind11')

    # [setups]
    sys.argv = [sys.argv[0]] + unknown

    setup(
        name='robotoy',
        version='0.2',
        description='A package for fast mathematical computations using C++ extensions.',
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        packages=setuptools.find_packages(),
        extras_require={},
        setup_requires=setup_requires,
        install_requires=install_requires,
        ext_modules=selected_extensions,
        cmdclass={'build_ext': CustomBuildExt},
        package_data={'robotoy': ['**/*.so']},  # IMPORTANT: Include .so files in package data
        zip_safe=False, # IMPORTANT: Set zip_safe to False
    )
