"""Week 6 Continuation: tuned CLF-DR-CBF and Random-CLF-DR-CBF on the canonical M0 map.

New code only. Nothing under dr_control/, robot_env/, evaluation/, optional_navigation/ or any
Week 5 / Week 5.5 path is edited; frozen modules are imported and wrapped, never modified.
See MD_files/week6_continuation/AUDIT.md for the design this package implements.
"""

import os as _os

# One BLAS thread per worker process: episodes are parallelised across processes instead.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
