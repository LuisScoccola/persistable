
A bit of theory
===============

.. _barcode:

----------------------------------------
The barcode of a hierarchical clustering
----------------------------------------

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
see Section 2 of 
`Stable and consistent density-based clustering <https://arxiv.org/abs/2005.09048>`__.

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


.. _persistence-based-flattening:

------------------------------------------
The persistence-based flattening algorithm
------------------------------------------

Given a hierarchical clustering of data, 
the persistence-based flattening algorithm extracts a single clustering 
of the data. We call this a "flattening" algorithm 
because it reduces a 1-parameter hierarchical clustering to 
a flat (0-parameter) clustering.

This tutorial provides a short description of the 
persistence-based flattening algorithm. 
For details, see Section 5 of 
`Stable and consistent density-based clustering <https://arxiv.org/abs/2005.09048>`__. 
This method is inspired by the ToMATo algorithm; 
see :ref:`links`.

The term "persistence" comes from the theory of persistent homology, 
and in particular, we will use the *barcode* of a hierarchical clustering 
to describe this algorithm:

.. figure:: pictures/hierarchical-clustering-barcode.png
    :align: center
    
    A schematic picture of a hierarchical clustering (in black), 
    and its barcode (in green).
    
See :ref:`barcode` for an introduction to this notion.

A simple flattening algorithm is to just take the *leaves* 
of the hierarchical clustering (the blue dashed edges):

.. figure:: pictures/hierarchical-clustering-leaves.png
    :align: center
    
Recall that bars in the barcode are created whenever a 
new cluster enters the hierarchy, 
so the bars are in one-to-one correspondence with the leaves 
of the hierarchical clustering. 
    
To be more explicit, 
let's add some data points to this picture:

.. figure:: pictures/hierarchical-clustering-all-points.png
    :align: center

The picture represents a hierarchical clustering of the set 
*{a,b,c,d,e}*. Point *a* enters the hierarchy at the parameter *r = 0*, 
and point *b* enters at *r=1*. 
At *r=1.5*, point *c* enters the hierarchy, and is immediately clustered with 
*a*. The clusters *{a,c}* and *{b}* merge at *r=2.5*, and so on.

The clustering of the data we get by taking the leaves has three clusters: 
*{a,c}*, *{b}*, and *{d}*. The point *e* is not assigned to any cluster, 
since it never lived inside a leaf.

A nice property of the leaf clustering is that it can identify cluster 
structure happening at different scales. 
In this example, once the point *d* has entered 
the hierarchy, points *a*, *b*, and *c* are all clustered together. 
So, we can't get the leaf clustering just by fixing a value of *r* 
and taking the hierarchical clustering at that value.

However, the leaf clustering can give poor results if the 
hierarchy has many spurious leaves, as in this example:
    
.. figure:: pictures/hierarchical-clustering-barcode-noise.png
    :align: center
    
The persistence-based flattening attempts to avoid this problem by 
first *pruning* the hierarchy, and then taking the leaf clustering. 
For this we use the barcode of the hierarchical clustering 
as a visual guide.

In the example above, we might guess that the three very short bars 
correspond to spurious leaves. 
So, we can choose a value *t* that is greater than the length of 
the three short bars, and prune the hierarchy by 
shortening each leaf by *t*:

.. figure:: pictures/hierarchical-clustering-barcode-noise-pruned.png
    :align: center
    
To see what happens to the data points inside the leaves, 
consider data points *a* and *b* living inside different leaves 
in the original hierarchical clustering: 

.. figure:: pictures/hierarchical-clustering-points.png
    :align: center
    
After pruning, these points are pushed rightwards along the hierarchy, 
and live inside the same leaf:
    
.. figure:: pictures/hierarchical-clustering-points-pruned.png
    :align: center    
    
Notice that, when we prune before taking the leaf clustering, 
we're not just removing low-persistence clusters from the end result. 
Since these low-persistence leaves can "interrupt" high-persistence 
leaves, pruning allows larger clusters to form.

As an example, consider the dataset from :ref:`introduction`. 
There, we considered a hierarchical clustering obtained 
by taking a slice of the DBSCAN graphs:

.. figure:: pictures/component_counting_slice.png
    :align: center
    
If we choose to keep 7 bars, 
the persistence-based flattening produces the following result:

.. figure:: pictures/hdbscan_data_7_clusters.png
    :align: center
     
If we choose to keep only 6 bars, 
the two clusters at the bottom are able to merge:

.. figure:: pictures/hdbscan_data_clustered.png
    :align: center
