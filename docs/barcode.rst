.. _barcode:

The barcode of a hierarchical clustering
========================================

Barcodes are a visualization tool for data analysis that 
play an important role in 
`persistent homology <https://www.ams.org/journals/bull/2009-46-02/S0273-0979-09-01249-X/>`_. 
This tutorial explains how barcodes 
provide a simple visualization of a hierarchical clustering.

.. figure:: pictures/hierarchical-clustering-barcode.png
    :align: center
    
.. |Hr| raw:: html

	<em>H(X)<sub>r</sub></em>

A (1-parameter) *hierarchical clustering* of a dataset *X* 
is a parameterized family of clusterings *H(X)* of *X*. 
Above (in black) is a schematic picture of a hierarchical clustering.

For real values *r* in some interval of the real line, we have a clustering |Hr| of the data, 
and as the parameter *r* grows, the clustering can change in the following ways:

1. A new cluster can be added to the hierarchy 
   (the triangles in the diagram).
2. Two clusters can merge (the dots).
3. The hierarchical clustering can end (the square).
4. New data points can be added to a cluster 
   (not shown in the diagram).
   
An example of such a hierarchical clustering is the output of the single-linkage algorithm. 
Persistable constructs hierarchical clusterings using graphs 
that encode both spatial relations among data points and density 
(see :ref:`introduction`).

For a rigorous definition of hierarchical clustering, 
see Section 2 
`here <https://arxiv.org/abs/2005.09048>`_.

The barcode of a hierarchical clustering *H(X)* consists of *bars*, 
which are simply intervals of the real line. 
In the picture above, the barcode is displayed in green. 
It can be constructed as follows: 

1. When a new cluster enters *H(X)* at parameter *r*, 
   start a new bar with left endpoint *r*.
2. If two clusters merge at parameter *r*, 
   take the cluster that entered the hierarchy at the 
   larger parameter value, and end its bar, 
   by setting the right endpoint to be *r*.
3. If the hierarchical clustering ends at parameter *r*, 
   set the right endpoint of all bars that 
   do not yet have a right endpoint to *r*.
   
Rule 2 is called the *Elder rule*, 
since the elder bar (that entered the hierarchy first), 
survives, while the younger bar dies.

For a rigorous description of this algorithm, see Definition 3.7 
`here <https://link.springer.com/article/10.1007/s41468-019-00024-z>`_.

.. 
   To do: update this link to our paper once we add 
   an algorithm for computing the barcode of a hierarchical clustering.

The barcode of a hierarchical clustering forgets some information. 
In particular, when two clusters merge, 
the barcode does not record which two clusters merged. 
This forgetfulness makes the barcode quite simple, 
which makes it easy to read even for fairly complicated hierarchical clusterings. 
See 
:ref:`introduction-clustering-with-persistable` 
to see how we use the barcode of a hierarchical clustering 
to choose parameters for Persistable.
