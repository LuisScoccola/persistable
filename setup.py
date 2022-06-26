from setuptools import setup
import warnings

try:
    from Cython.Distutils import build_ext
    from setuptools import setup, Extension
    HAVE_CYTHON = True

    from Cython.Compiler.Options import get_directive_defaults
    directive_defaults = get_directive_defaults()
    directive_defaults['linetrace'] = True
    directive_defaults['binding'] = True

except ImportError as e:
    warnings.warn(e.args[0])
    from setuptools import setup, Extension
    from setuptools.command.build_ext import build_ext
    HAVE_CYTHON = False

def requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip()]

class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""

    def run(self):
        import numpy
        self.include_dirs.append(numpy.get_include())
        build_ext.run(self)

aux = Extension('linkage.aux', sources=['linkage/aux.pyx'], define_macros=[('CYTHON_TRACE', '1')])
relabel_dendrogram = Extension('linkage.borrowed.relabel_dendrogram',
                             sources=['linkage/borrowed/relabel_dendrogram.pyx'],
                             define_macros=[('CYTHON_TRACE', '1')])
dist_metrics = Extension('linkage.borrowed.dist_metrics',
                         sources=['linkage/borrowed/dist_metrics.pyx'],
                             define_macros=[('CYTHON_TRACE', '1')])
_hdbscan_boruvka = Extension('linkage.borrowed._hdbscan_boruvka',
                             sources=['linkage/borrowed/_hdbscan_boruvka.pyx'],
                             define_macros=[('CYTHON_TRACE', '1')])

if not HAVE_CYTHON:
    raise ImportError('Cython not found!')

setup(
   name='linkage',
   version='0.1',
   description='Implements the gamma-linkage hierarchical clustering algorithm of (Rolle and Scoccola, 2021)',
   license='3-clause BSD',
   maintainer='Luis Scoccola',
   maintainer_email='luis.scoccola@gmail.com',
   packages=['linkage'],
   install_requires=requirements(),
   ext_modules=[relabel_dendrogram, aux, dist_metrics, _hdbscan_boruvka],
   cmdclass={'build_ext': CustomBuildExtCommand},
   data_files=('linkage/borrowed/dist_metrics.pxd',),
)
