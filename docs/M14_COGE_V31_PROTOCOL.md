# COGE V3.1 Protocol

Calibrated Opportunity-Gated Evolution (COGE) separates two decisions:

1. **Opportunity:** is there evidence that any safe persistent update improves
   demand-weighted utility over `HOLD` by at least a predeclared margin?
2. **Intervention:** if so, choose the smallest supported grammar update;
   otherwise choose `HOLD`.

All non-HOLD updates must satisfy paired reach and pick-place retention within
0.05 of the measured `HOLD` capability. A controller outside the calibration
support or without a positive calibrated lower bound abstains.

V3.1 adds micro-updates (25, 50, 100, and 250 gradient steps) so the governor
can attempt a bounded intervention before a 500--1,000-step update. The
opportunity gate, grammar, margins, and calibration split must be frozen before
training a new blind controller bank.
