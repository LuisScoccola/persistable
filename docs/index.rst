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
Installation through PyPI is coming soon.

.. code:: bash
    pip install git+https://github.com/LuisScoccola/persistable.git

Persistable depends on the following python packages:
`numpy`, `scipy`, `scikit-learn`, `cython`, `plotly`, `dash`, `jupyter_dash`, `diskcache`, `multiprocess`, `psutil`.


Contents
--------

.. toctree::
    :maxdepth: 2

    quickstart
    introductiontopersistable
    abitoftheory 
    links
    api
