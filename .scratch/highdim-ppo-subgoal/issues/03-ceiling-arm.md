# 03: Ceiling arm: A* + Random through the same harness

**What to build:** The researcher can run the ceiling arm (frozen A* + carrot + Random-CLF-DR-CBF, i.e. the F-D arm) through the same episode harness, and it is provably the frozen baseline.

**Blocked by:** 02 (Floor arm tracer)

**Status:** ready-for-agent

- [ ] The ceiling arm is built by delegating to the existing Continuation construction of the Random arm and its recovery attachment, verbatim
- [ ] Seam-1 test: ceiling trajectories are bit-identical to the Continuation harness's Random arm on 5 existing validation seeds
- [ ] Seam-1 test: the same seed reproduces an identical ceiling episode
- [ ] Running both arms on the same seeds asserts identical start and goal across arms
- [ ] No frozen file is edited, and the manifest test still passes
