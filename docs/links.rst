.. _links:

Related tools for density-based clustering
==========================================

The theory behind Persistable was developed in the paper
`Stable and consistent density-based clustering <https://arxiv.org/abs/2005.09048>`_. 
The introduction to the paper describes the previous work 
by many researchers that we are building on.

This page provides links to some clustering tools that 
users of Persistable may also be interested in.

* `hdbscan <https://github.com/scikit-learn-contrib/hdbscan>`_. 
  A high performance implementation of HDBSCAN clustering. The hierarchical clustering method used by HDBSCAN is closely related to the hierarchies used by Persistable, and this implementation of Persistable uses the high-performance algorithms for hierarchical clustering implemented here.
* `RIVET <https://github.com/rivetTDA/rivet>`_. 
  A tool for 2-parameter persistent homology. In particular, this software implements tools for visualizing the *degree-Rips* persistent homology of a finite metric space. This is closely related to the visualization of the component counting function provided by Persistable, though we use different computational methods. The visualization tools provided by RIVET provide an even more fine-grained view of the DBSCAN graphs.
* `ToMATo, implemented in the GUDHI library <https://gudhi.inria.fr/python/latest/clustering.html>`_. 
  ToMATo is another approach to density-based clustering. This was the first algorithm to use persistence to choose the number of clusters.
