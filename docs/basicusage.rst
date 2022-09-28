Basic usage
===========

Under construction.

.. code:: python

    import persistable
    from sklearn.datasets import make_blobs

    X = make_blobs(2000, random_state=1)[0]
    p = persistable.Persistable(X)
    clustering_labels = p.quick_cluster()
