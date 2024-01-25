Test performance of TeNPy's add_op function for large local bases.

Uses the `1D-SU3-2flavor-2level` lattice gauge model from the [QHLGT-models](https://baltig.infn.it/qpd/qhlgt-models) repository.

__requirements:__
`physics-tenpy`; the conda precise environment where I did the testing is in `environment.yml`.

__usage:__
`git clone --recursive` and run `python tenpy_1d_lgt.py` to cProfile the model initialization and save the results in a `prof` file.

__license__: GPLv3.
