from setuptools import setup, Extension

module = Extension(
    "mykmeanssp",
    sources=["kmeansmodule.c"]
)

setup(
    name="mykmeanssp",
    version="1.0",
    description="K-means C extension module",
    ext_modules=[module]
)