---
title: 'Persistable: persistent and stable clustering'
tags:
  - clustering
  - unsupervised learning
  - machine learning
  - topological data analysis
authors:
 - name: Luis Scoccola 
   orcid: 0000-0002-4862-722X
   affiliation: 1
 - name: Alexander Rolle
   affiliation: 2
affiliations:
 - name: Northeastern University
   index: 1
 - name: Technical University of Munich 
   index: 2
date: 15 November 2022
bibliography: paper.bib
---

# Summary

Persistable is an implementation of density-based clustering algorithms intended for exploratory data analysis.
What distinguishes Persistable from other clustering software is its visualization capabilities.
Persistable's interactive mode lets the user visualize multi-scale and multi-density cluster structure present in the data.
This is used to guide the choice of parameters that lead to the final clustering.

Persistable is based on multi-parameter persistence [@botnanlesnick], a method from topological data analysis; the theory behind Persistable is developed in [@rollescoccola].
Persistable is implemented in Python, with the most expensive computations---in particular the implementations borrowed from [@scipy] and from the high performance algorithms for density-based clustering developed in [@mcinneshealy] and implemented in [@mcinneshealyastels]---done in Cython.
Persistable's interactive mode is inspired by RIVET [@rivet] and is implemented in Plotly Dash [@plotly].
We test the core algorithms as well as the graphical user interface in Ubuntu, macOS, and Windows.
We have designed Persistable with the goal of making both its computational components and its GUI easy to extend.
We hope to keep adding high performance topological inference functionality.

The documentation for Persistable can be found at [persistable.readthedocs.io](https://persistable.readthedocs.io/).

# Related work and statement of need

There exist many implementations of density-based clustering algorithms.
Several of these are related to and serve some of the same purposes as Persistable.
The main novel contribution of Persistable is that of providing visualization tools based on persistence which inform the selection of all parameters.
We review existing algorithms and implementations that are related to Persistable, and briefly describe Persistable's main functionality.

**DBSCAN.**
The algorithm was introduced in [@dbscan], and an implementation is available at [@scikit-learn].
DBSCAN takes two parameters, a scale parameter and a density threshold, which are used to construct a graph on the data.
This graph models the connectivity properties of the data, with respect to the chosen parameters, and the DBSCAN clustering is the set of components of this graph.
A main advantage of DBSCAN is that the output clustering is very interpretable in terms of the chosen parameters; meanwhile, a major difficulty for practitioners is the choice of these parameters, especially without a priori knowledge of the scale of the data (see, e.g., [@dbscan-2]).

**HDBSCAN.**
The algorithm was introduced in [@campello2013density], and a high performance implementation is in [@mcinneshealyastels].
HDBSCAN can be seen as a hierarchical version of DBSCAN, which eliminates the dependence on the scale parameter by considering the hierarchical clustering defined by fixing a density threshold, and letting the distance scale vary.
The algorithm extracts a single clustering from this hierarchy according to a certain notion of persistence, using an additional minimum cluster size parameter.
Eliminating the scale parameter makes HDBSCAN easier to use than DBSCAN in many cases, but choosing the density threshold can still be a challenge for practitioners.
The implementation of [@mcinneshealyastels] has visualization functionality which aides in the choice of minimum cluster size.

**ToMATo.**
The algorithm was introduced in [@chazaletal] and is implemented in the GUDHI library [@gudhi].
This was the first clustering algorithm to use persistence to do parameter selection, specifically by using a persistence diagram to choose the number of clusters.
The basic input for ToMATo is a graph and a density estimate.
How these are chosen is left to the user, and their choice is not informed by persistence.
<!--; this stands in contrast to Persistable, which is based on similar procedures but offers visualizations that aid in the selection of all parameters.-->

**RIVET.**
The theory behind RIVET is developed in [@lesnick-wright-2] and [@lesnick-wright], and the official implementation is in [@rivet].
This is a very general tool for 2-parameter persistent homology, and can, in particular, produce visualizations that are very similar to the ones produced by Persistable.
Indeed, we have taken RIVET as inspiration when designing Persistable.
However, RIVET is not primarily intended to be clustering software, and it does not produce a clustering of the data.
<!--(Mention scalability?)-->

**Persistable.**
We contribute to this landscape of density-based clustering methods by providing a clustering pipeline in which every choice of parameter is guided by visualization tools.
Persistable first builds a hierarchical clustering of data, in a way that is similar to HDBSCAN: we take a one-parameter family of DBSCAN\* graphs^[DBSCAN\* is a slight modification of DBSCAN introduced in [@campello2013density], which removes the notion of border point, simplifying the theory and implementation without compromising efficacy.] by choosing a line through the DBSCAN\* parameter space.
This choice is guided by a *component counting function* plot, which is a summary of the output of DBSCAN\* for a large set of parameters.
We then extract a single clustering using persistence, similar to ToMATo.
This choice is guided by a *prominence vineyard*.


# Example

We use a synthetic dataset from [@mcinneshealy] and [@mcinneshealyastels], which is challenging for most clustering algorithms but easy to visualize.
As shown in Figure \ref{figure:gui}, Persistable's visualizations suggest that there are six clusters that persist across several distance scales as well as several density thresholds.
We show in Figure \ref{figure:output} the flat clustering obtained by selecting those six high-prominence clusters.

It is worth pointing out that the component counting function of Figure \ref{figure:gui} summarizes 10,000 runs of DBSCAN\* on a dataset with a bit more than 2000 points, and takes about 2 seconds to run on a MacBook Pro with M1 chip and a 10-Core CPU.

![Using Persistable's GUI to select parameters. We look at a family of 100 one-parameter hierarchical clustering that are restrictions of the two-parameter hierarchical clustering obtained by running DBSCAN\* with all possible parameters.\label{figure:gui}](GUI.png)

![Output clustering obtained with the parameters of Figure \ref{figure:gui}. Grey points do not belong to any cluster.\label{figure:output}](clustered-data.png){width=50%}


# Acknowledgements

We thank Leandro Lovisolo and Manuel Ferreria for several fruitful conversations.


# References
