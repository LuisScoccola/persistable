.. image:: pictures/logo.svg
  :width: 550
  :align: center

|

Persistent and stable clustering (Persistable) is a density-based clustering algorithm intended for exploratory data analysis.
What distinguishes Persistable from other clustering algorithms is its visualization capabilities.
Persistable's interactive mode lets you visualize multi-scale and multi-density cluster structure present in the data.
This is used to guide the choice of parameters that lead to the final clustering.


Installing
----------

Make sure you are using Python 3.
Persistable depends on the following python packages, which will be installed automatically when you install with `pip`:
`numpy`, `scipy`, `scikit-learn`, `cython`, `plotly`, `dash`, `diskcache`, `multiprocess`, `psutil`.
To install from pypi, simply run the following:

.. code-block::

    pip install persistable-clustering


Contents
--------

.. toctree::
    :maxdepth: 2

    quickstart
    introductiontopersistable
    abitoftheory 
    links
    api
