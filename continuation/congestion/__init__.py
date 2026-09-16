"""CD-random-CLF-DR-CBF: congestion detection on top of the frozen Random-CLF-DR-CBF.

signals.py  diagnostic signals (free gap, track proximity / closing motion)
monitor.py  grid feasibility margin M - tau_eff at the current and predicted states
detector.py Normal / Cautious / Critical with hysteresis; alarm statistics
wrapper.py  nominal-action + CLF-target caution wrapper, installed BEFORE Random recovery
"""
