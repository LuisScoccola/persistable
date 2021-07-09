# gamma-linkage v0.1

Implements the gamma-linkage hierarchical clustering algorithm of [1].
The algorithm generalizes the HDBSCAN algorithm [2] and other standard density-based hierarchical clustering algorithms, and enjoys better stability properties (see [1, Sections 3 and 5]).
Please see the jupyter notebooks for examples.

## Authors

Luis Scoccola and Alexander Rolle.

## Next version

The current version (v0.1) is relatively efficient, but is quadratic in the number of points.
We are currently working on a subquadratic implementation.

## References

    [1] Stable and consistent density-based clustering. A. Rolle and L. Scoccola. https://arxiv.org/abs/2005.09048, 2021.

    [2] Density-based clustering based on hierarchical density estimates. R. J. G. B. Campello, D. Moulavi, and J. Sander. Advances in Knowledge Discovery and Data Mining, volume 7819 of Lecture Notes in Computer Science, pp. 160-172. Springer, 2013.

## License

The software is published under the 3-clause BSD license.
