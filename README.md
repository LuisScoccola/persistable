
[![PyPI](https://img.shields.io/pypi/v/persistable-clustering?color=green)](https://pypi.org/project/persistable-clustering)
[![Downloads](https://static.pepy.tech/personalized-badge/persistable-clustering?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads)](https://pepy.tech/project/persistable-clustering)
[![tests](https://github.com/LuisScoccola/persistable/actions/workflows/run_tests.yaml/badge.svg)](https://github.com/LuisScoccola/persistable/actions/workflows/run_tests.yaml)
[![coverage](https://codecov.io/gh/LuisScoccola/persistable/branch/main/graph/badge.svg)](https://codecov.io/gh/LuisScoccola/persistable)
[![docs](https://readthedocs.org/projects/persistable/badge/?version=latest)](https://persistable.readthedocs.io/)
[![status](https://joss.theoj.org/papers/63d612cd4730c3aa708e3a47eb2c50f3/status.svg)](https://joss.theoj.org/papers/63d612cd4730c3aa708e3a47eb2c50f3)
[![license](https://img.shields.io/github/license/LuisScoccola/persistable)](https://github.com/LuisScoccola/persistable/blob/main/LICENSE)
---

<p align="center">
    <img src="https://raw.githubusercontent.com/LuisScoccola/persistable/main/docs/pictures/logo.svg" width="550">
</p>

Persistent and stable clustering (Persistable) is a density-based clustering algorithm intended for exploratory data analysis.
What distinguishes Persistable from other clustering algorithms is its visualization capabilities.
Persistable's interactive mode lets you visualize multi-scale and multi-density cluster structure present in the data.
This is used to guide the choice of parameters that lead to the final clustering.


## Usage

Here is a brief outline of the main functionality; see the [documentation](https://persistable.readthedocs.io/) for details, including the [API reference](https://persistable.readthedocs.io/en/latest/api.html).

In order to run Persistable's interactive mode from a Jupyter notebook, run the following in a Jupyter cell:

```python
import persistable
from sklearn.datasets import make_blobs

X = make_blobs(2000, centers=4, random_state=1)[0]

p = persistable.Persistable(X)
pi = persistable.PersistableInteractive(p)
pi.start_ui()
```

The last command returns the port in `localhost` serving the UI, which is `8050` by default.
Now go to `localhost:8050` in your web browser to access the graphical user interface:

![Alt text](https://raw.githubusercontent.com/LuisScoccola/persistable/main/docs/pictures/GUI.png)

After choosing your parameters using the user interface, you can get your clustering in another Jupyter cell by running:

```python
clustering_labels = pi.cluster()
```

**Note:** You may use `pi.start_ui(jupyter_mode="inline")` to have the graphical user interface display directly in the Jupyter notebook!


## Installing

Make sure you are using Python 3.
Persistable depends on the following python packages, which will be installed automatically when you install with `pip`:
`numpy`, `scipy`, `scikit-learn`, `cython`, `plotly`, `dash`, `diskcache`, `multiprocess`, `psutil`.
To install from pypi, simply run the following:

```bash
pip install persistable-clustering
```


## Documentation and support

You can find the documentation at [persistable.readthedocs.io](https://persistable.readthedocs.io/).
If you have further questions, please [open an issue](https://github.com/LuisScoccola/persistable/issues/new) and we will do our best to help you.
Please include as much information as possible, including your system's information, warnings, logs, screenshots, and anything else you think may be of use.
If you do not wish to open an issue, you are also welcome to contact [Luis Scoccola](https://luisscoccola.github.io/) directly.
Please be patient if it takes us a bit to get back to you.



## Running the tests

You can run the tests by running the following commands from the root directory of a clone of this repository.
If a test fails, please [report a bug](https://github.com/LuisScoccola/persistable/issues/new), trying to include as much information as possible, including your system's information, warnings, logs, screenshots, and anything else you think may be of use.

```bash
pip install pytest playwright pytest-playwright
python -m playwright install --with-deps
pip install -r requirements.txt
python -m setup build_ext --inplace
pytest .
```


## Details about theory and implementation

Persistable is based on multi-parameter persistence [[4]](#4), a method from topological data analysis.
The theory behind Persistable is developed in [[1]](#1), while this implementation uses the high performance algorithms for density-based clustering developed in [[2]](#2) and implemented in [[3]](#3).
Persistable's interactive mode is inspired by RIVET [[5]](#5) and is implemented in [Dash](https://dash.plotly.com/).


## Contributing

To contribute, you can fork the project, make your changes, and submit a pull request.
You may want to contact [Luis Scoccola](https://luisscoccola.github.io/) first, to make sure your work does not overlap with ongoing work.


## Authors

[Luis Scoccola](https://luisscoccola.github.io/) and [Alexander Rolle](https://alexanderrolle.github.io/).

## Citing

If you use this package in your work, you may cite the corresponding paper using the following bibtex entry:

```
@article{Scoccola2023,
    doi = {10.21105/joss.05022},
    url = {https://doi.org/10.21105/joss.05022},
    year = {2023},
    publisher = {The Open Journal},
    volume = {8},
    number = {83},
    pages = {5022},
    author = {Luis Scoccola and Alexander Rolle},
    title = {Persistable: persistent and stable clustering},
    journal = {Journal of Open Source Software}
}
```

## References

<a id="1">[1]</a> 
*Stable and consistent density-based clustering*. A. Rolle and L. Scoccola. [arXiv:2005.09048](https://arxiv.org/abs/2005.09048)

<a id="2">[2]</a> 
*Accelerated Hierarchical Density Based Clustering*. L. McInnes, J. Healy. 2017 IEEE International Conference on Data Mining Workshops (ICDMW), IEEE, pp 33-42. 2017

<a id="3">[3]</a> 
*hdbscan: Hierarchical density based clustering*. L. McInnes, J. Healy, S. Astels. Journal of Open Source Software, The Open Journal, volume 2, number 11. 2017

<a id="4">[4]</a> 
*An Introduction to Multiparameter Persistence*. M. B. Botnan, M. Lesnick. Proceedings of the 2020 International Conference on Representations of Algebras. 2022

<a id="5">[5]</a> 
*RIVET*. The RIVET Developers. [[Git]](https://github.com/rivetTDA/rivet) [[docs]](https://rivet.readthedocs.io/en/latest/index.html)

<!---
<a id="4">[4]</a> 
*Density-based clustering based on hierarchical density estimates*. R. J. G. B. Campello, D. Moulavi, and J. Sander. Advances in Knowledge Discovery and Data Mining, volume 7819 of Lecture Notes in Computer Science, pp. 160-172. Springer, 2013.
-->


## License

This software is published under the 3-clause BSD license.
