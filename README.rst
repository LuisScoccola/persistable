===================================================
Persistable: persistent and stable clustering (v0.2)
===================================================

Implements the lambda-linkage hierarchical clustering algorithm and the persistence-based flattening of [1].
The algorithm generalizes the HDBSCAN algorithm [2] and other standard density-based hierarchical clustering algorithms, and enjoys better stability properties (see [1, Sections 3 and 5]).
Please see the Jupyter notebooks for examples.

----------
Installing
----------

.. code:: bash

    pip install --upgrade git+https://github.com/LuisScoccola/persistable.git@fast-mst

-------
Authors
-------

Luis Scoccola and Alexander Rolle.

----------
References
----------

    [1] Stable and consistent density-based clustering. A. Rolle and L. Scoccola. https://arxiv.org/abs/2005.09048, 2021.

    [2] Density-based clustering based on hierarchical density estimates. R. J. G. B. Campello, D. Moulavi, and J. Sander. Advances in Knowledge Discovery and Data Mining, volume 7819 of Lecture Notes in Computer Science, pp. 160-172. Springer, 2013.

-------
License
-------

The software is published under the 3-clause BSD license.
