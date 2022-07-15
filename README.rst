=============================================
Persistable: persistent and stable clustering
=============================================

High performance implementation of Persistable clustering, a combination of the
lambda-linkage hierarchical clustering algorithm and the persistence-based
flattening of [1]. Persistable is a density-based clustering algorithm
intended for exploratory data analysis.

The algorithm is similar in spirit to the HDBSCAN algorithm [2] and other
standard density-based hierarchical clustering algorithms, but enjoys better
stability properties (see [1, Sections 3 and 5]). This implementation is based
on the high performance algorithms for density-based clustering developed
in [3] and implemented in [4].


Usage
-----

We are currently working on the documentation.
For now, please see the Jupyter notebooks for examples.
Please keep in mind that this is a beta version and the user interface may
change with the stable release.


Installing
----------

.. code:: bash

    pip install git+https://github.com/LuisScoccola/persistable.git


Authors
-------

Luis Scoccola and Alexander Rolle.


References
----------

    [1] Stable and consistent density-based clustering. A. Rolle and L. Scoccola. https://arxiv.org/abs/2005.09048, 2021.

    [2] Density-based clustering based on hierarchical density estimates. R. J. G. B. Campello, D. Moulavi, and J. Sander. Advances in Knowledge Discovery and Data Mining, volume 7819 of Lecture Notes in Computer Science, pp. 160-172. Springer, 2013.

    [3] Accelerated Hierarchical Density Based Clustering. McInnes L, Healy J. 2017 IEEE International Conference on Data Mining Workshops (ICDMW), IEEE, pp 33-42. 2017

    [4] hdbscan: Hierarchical density based clustering. L. McInnes, J. Healy, S. Astels. Journal of Open Source Software, The Open Journal, volume 2, number 11. 2017


License
-------

The software is published under the 3-clause BSD license.
