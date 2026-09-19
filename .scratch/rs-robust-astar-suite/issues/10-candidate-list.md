# 10: Candidate list

**What to build:** The researcher gets at most three written improvement candidates, grounded in the diagnostic, and approves them before any candidate is built.

**Blocked by:** 09

**Status:** ready-for-agent

- [ ] Each candidate is written down: mechanism, changed components, information used (onboard only), expected effect on the named failure modes, parameter-search budget, and an expected compute cost
- [ ] At most three candidates; a combination of mechanisms may be one of them
- [ ] No candidate changes the CBF/DR formulation or the dynamics, or uses ground-truth or trigger/spawn information
- [ ] The list is recorded in the design document before any tuning run
- [ ] **The researcher approves the list.** Ticket 11 is then replaced by one implementation ticket per approved candidate
