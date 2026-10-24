from setuptools import setup, Extension

module = Extension(
    "symnmf",
    sources=["symnmfmodule.c", "symnmf.c"],
    extra_compile_args=["-DFROM_MODULE"]
)

setup(
    name="symnmf",
    version="1.0",
    ext_modules=[module]
)