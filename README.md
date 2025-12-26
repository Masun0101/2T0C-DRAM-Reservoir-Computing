# 2T0C-DRAM-Reservoir-Computing
2T0C DRAM  Reservoir Computing Simulation Platform
# MicroSpice: 2T0C DRAM-based Reservoir Computing Framework

## 📖 Overview

This repository contains the source code and simulation framework for the research project: **"Spatio-Temporal Information Processing using Reaction-Diffusion Dynamics in 2T0C DRAM Arrays."**

We introduce **MicroSpice**, a custom lightweight circuit solver developed in Python, designed specifically to simulate the non-linear dynamics of IGZO-based 2T0C (2-Transistor-0-Capacitor) DRAM cells. By exploiting the intrinsic leakage and capacitive coupling of 2T0C units, this framework implements a physical Reservoir Computing (RC) system capable of processing complex temporal tasks.

### Key Components
1.  **MicroSpice Solver:** A custom discrete-time circuit simulator optimized for large-scale dynamical networks. It models IGZO TFT sub-threshold leakage, dynamic threshold voltage shift ($\Delta V_{th}$), and node-to-node coupling.
2.  **RC Benchmarks:** Implementation of standard Reservoir Computing tasks, including the **Multiple Superimposed Oscillator (MSO)** task, to validate the computing capability of the 2T0C array.

---

## 🚀 Features

* **Physics-Informed Modeling:** Accurate behavioral modeling of IGZO TFTs, capturing critical leakage current behaviors essential for fading memory properties.
* **Reaction-Diffusion Dynamics:** Simulates the "Reaction" (via 2T0C non-linear integration) and "Diffusion" (via resistive coupling) mechanisms within the array.
* **Customizable Topology:** Allows users to define 1D chains or 2D grids of 2T0C cells with tunable coupling strengths.
* **Standard Benchmarks:** Includes pre-configured scripts for MSO (Multiple Superimposed Oscillator) and other time-series prediction tasks.
