---
title: HH-MPNN ACOPF Decision Support Tool
---

# HH-MPNN ACOPF Decision Support Tool

Streamlit dashboard serving precomputed ACOPF surrogate inference from a
Hybrid Heterogeneous Message Passing Neural Network (HH-MPNN) on the
IEEE 118-bus system. Read our paper here: https://www.sciencedirect.com/science/article/pii/S2666546826001680

- **Load data**: realistic yearly load profile from the Romanian grid,
  scaled for the IEEE-118 bus grid — 35040 scenarios at 15-minute resolution.
- **Grid & Dispatch**: predicted generator dispatch (green) on the 118-bus
  graph; branches violating their rating drawn in red.
- **Violations**: power balance mismatches (MW / MVar) and branch flow
  violations (MVA).
- **Cost**: dispatch cost of the predicted operating point in dollars.

The app replays the precomputed year in order, advancing one scenario every
15 minutes of wall-clock time (a demo-speed toggle in the sidebar advances
every 10 seconds instead).

All heavy computation (GNN inference + physics evaluation over the full
year) was done offline; this Space only serves the results file
`results_118_year.npz`.
