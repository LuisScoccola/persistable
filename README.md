# Persistable: persistent and stable clustering

Persistable is a density-based clustering algorithm intended for exploratory data analysis.
What distinguish Persistable from other clustering algorithms are its visualization capabilities.
Persistable's interactive mode lets the user visualize muti-scale as well as multi-density cluster structure present in the data.
This structure is intrinsic to the data and independent of parameter choices, and is used to inform the choice of parameters needed to obtain the final clustering.


## Usage

Please be patient while we work on the documentation.
Here is a brief outline of the main functionality.
Please keep in mind that this is a beta version and the user interface may change with the stable release.

### Basic usage

For further examples, please see the Jupyter notebooks.

```python
import persistable
from sklearn.datasets import make_blobs

X = make_blobs(2000)[0]
p = persistable.Persistable(X)
clustering_labels = p.quick_cluster()
```


### Interactive mode 

This is where the real power of Persistable is.

#### From a Jupyter notebook

For now, interactive mode is only supported through Jupyter notebooks.
In one cell, run:

```python
import persistable
from sklearn.datasets import make_blobs

X = make_blobs(2000)[0]
p = persistable.Persistable(X)
pi = persistable.PersistableInteractive(p, jupyter = True, inline = False)
```

Now go to `localhost:8050` in your browser to interact with the data.
After choosing your parameters by clicking on the "Choose parameters" button in the GUI, you can get your clustering in another cell with.

```python
cluster_labels = p.cluster(**pi._parameters)
```

Note: you may use `inline = True` to have the GUI run directly in the Jupyter notebook instead of the browser!


<!---
#### From a Python script

```python
import persistable
from sklearn.datasets import make_blobs

X = make_blobs(2000)[0]
p = persistable.Persistable(X)
pi = persistable.PersistableInteractive(p)
# will wait until you close the GUI
cluster_labels = p.cluster(**pi._parameters)
```

This will run the lines up to the commented line, and it will wait for you to interact with the data.
Now go to `localhost:8050` in your browser to interact with the data.
You can then fix your chosen parameters and close the app by clicking on the "Choose parameters and close" button in the GUI.
--->

## Installing

```bash
pip install git+https://github.com/LuisScoccola/persistable.git
```


## Details about theory and implementation

Persistable is based on multi-parameter persistence [[4]](#4), a notion from Topological Data Analysis.
The theory behind Persistable is developed in [[1]](#1), while this implementation uses the high performance algorithms for density-based clustering developed in [[2]](#2) and implemented in [[3]](#3).


## Authors

Luis Scoccola and Alexander Rolle.


## References

<a id="1">[1]</a> 
*Stable and consistent density-based clustering*. A. Rolle and L. Scoccola.
https://arxiv.org/abs/2005.09048

<a id="2">[2]</a> 
*Accelerated Hierarchical Density Based Clustering*. McInnes L, Healy J. 2017 IEEE International Conference on Data Mining Workshops (ICDMW), IEEE, pp 33-42. 2017

<a id="3">[3]</a> 
*hdbscan: Hierarchical density based clustering*. L. McInnes, J. Healy, S. Astels. Journal of Open Source Software, The Open Journal, volume 2, number 11. 2017

<a id="4">[4]</a> 
*An Introduction to Multiparameter Persistence*. Magnus Bakke Botnan, Michael Lesnick.
https://arxiv.org/abs/2203.14289

<!---
<a id="4">[4]</a> 
*Density-based clustering based on hierarchical density estimates*. R. J. G. B. Campello, D. Moulavi, and J. Sander. Advances in Knowledge Discovery and Data Mining, volume 7819 of Lecture Notes in Computer Science, pp. 160-172. Springer, 2013.
-->


## License

The software is published under the 3-clause BSD license.
