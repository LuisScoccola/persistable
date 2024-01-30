import warnings
import sys

# We only want linetrace for profiling cython when running coverage tests, and
# do not compile with this option in all other cases.
# The following may not be the ideal way to do this, but it works for now.
# We may want to switch to something like:
# https://docs.python.org/3/distutils/apiref.html#creating-a-new-distutils-command
if "--compile-with-cython-linetrace" in sys.argv:
    cython_trace = True
    sys.argv.remove("--compile-with-cython-linetrace")
else:
    cython_trace = False


try:
    from Cython.Distutils import build_ext
    from setuptools import setup, Extension

    HAVE_CYTHON = True

    if cython_trace:
        from Cython.Compiler.Options import get_directive_defaults

        directive_defaults = get_directive_defaults()
        directive_defaults["linetrace"] = True
        directive_defaults["binding"] = True

except ImportError as e:
    warnings.warn(e.args[0])
    from setuptools import setup, Extension
    from setuptools.command.build_ext import build_ext

    HAVE_CYTHON = False


def requirements():
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip()]


class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""

    def run(self):
        import numpy

        self.include_dirs.append(numpy.get_include())
        build_ext.run(self)


if cython_trace:
    define_macros = [("CYTHON_TRACE_NOGIL", "1")]
else:
    define_macros = []


auxiliary = Extension(
    "persistable.auxiliary",
    sources=["persistable/auxiliary.pyx"],
    define_macros=define_macros,
)
persistence_diagram_h0 = Extension(
    "persistable.persistence_diagram_h0",
    sources=["persistable/persistence_diagram_h0.pyx"],
    define_macros=define_macros,
)
signed_betti_numbers = Extension(
    "persistable.signed_betti_numbers",
    sources=["persistable/signed_betti_numbers.pyx"],
    define_macros=define_macros,
)
subsampling = Extension(
    "persistable.subsampling",
    sources=["persistable/subsampling.pyx"],
    define_macros=define_macros,
)
dense_mst = Extension(
    "persistable.borrowed.dense_mst",
    sources=["persistable/borrowed/dense_mst.pyx"],
    define_macros=define_macros,
)
dist_metrics = Extension(
    "persistable.borrowed.dist_metrics",
    sources=["persistable/borrowed/dist_metrics.pyx"],
    define_macros=define_macros,
)
prim_mst = Extension(
    "persistable.borrowed.prim_mst",
    sources=["persistable/borrowed/prim_mst.pyx"],
    define_macros=define_macros,
)
_hdbscan_boruvka = Extension(
    "persistable.borrowed._hdbscan_boruvka",
    sources=["persistable/borrowed/_hdbscan_boruvka.pyx"],
    define_macros=define_macros,
)

if not HAVE_CYTHON:
    raise ImportError("Cython not found!")

def readme():
    with open("README.md") as readme_file:
        return readme_file.read()

setup(
    name="persistable-clustering",
    version="0.5.1",
    description="Density-based clustering for exploratory data analysis based on multi-parameter persistence",
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved",
        "Programming Language :: Python",
        "Programming Language :: C",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ],
    keywords="clustering density hierarchical persistence TDA",
    url="https://github.com/LuisScoccola/persistable",
    license="3-clause BSD",
    maintainer="Luis Scoccola",
    maintainer_email="luis.scoccola@gmail.com",
    packages=["persistable"],
    install_requires=requirements(),
    ext_modules=[
        signed_betti_numbers,
        auxiliary,
        persistence_diagram_h0,
        subsampling,
        dense_mst,
        dist_metrics,
        prim_mst,
        _hdbscan_boruvka,
    ],
    cmdclass={"build_ext": CustomBuildExtCommand},
    data_files=(
        "persistable/borrowed/dist_metrics.pxd",
        "persistable/borrowed/_hdbscan_boruvka.pxd",
    ),
    include_package_data=True,
)
