"""Phase 5 global-navigation layer, kept separate from dr_control/ on purpose.

Phases 1-4 validated the LOCAL controller and must remain runnable without any of this, so
nothing in dr_control/ imports this package. Phase 5 adds a global reference only:

    A* over the KNOWN static map  ->  waypoint (carrot) following  ->  CLF  ->  DR-CBF

The reference governor (plan 5a) and online occupancy mapping (plan 5b, an extension) are
NOT part of this step.
"""
