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

graph LR
    %% 定义样式
    classDef input fill:#f9f,stroke:#333,stroke-width:2px;
    classDef dram fill:#d4e1f5,stroke:#333,stroke-width:2px;
    classDef coupling fill:#ffcccc,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5;
    classDef readout fill:#e1d5e7,stroke:#333,stroke-width:2px;
    classDef output fill:#d5e8d4,stroke:#333,stroke-width:2px;

    %% 输入层
    subgraph Input_Layer [Time-Series Input]
        In(u(t)):::input
    end

    %% 储备池层 (2T0C Array)
    subgraph Reservoir_Layer [2T0C Reaction-Diffusion Reservoir]
        direction LR
        
        %% 物理节点
        N1((Node 1<br>2T0C)):::dram
        N2((Node 2<br>2T0C)):::dram
        N3((Node 3<br>2T0C)):::dram
        N4((Node ...<br>2T0C)):::dram
        
        %% 扩散耦合 (电阻)
        N1 <-->|R_diff| N2
        N2 <-->|R_diff| N3
        N3 <-->|R_diff| N4
        
        %% 反应项说明 (隐藏线条，仅做标注)
        N1 -.- |Leakage &<br>Integration| N1
    end

    %% 读出层
    subgraph Readout_Layer [Linear Readout]
        Sum((Σ)):::readout
    end

    %% 输出
    Out(y(t)):::output

    %% 连接关系
    In -->|Masking| N1
    In -->|Masking| N2
    In -->|Masking| N3
    In -->|Masking| N4

    N1 -->|w1| Sum
    N2 -->|w2| Sum
    N3 -->|w3| Sum
    N4 -->|...| Sum

    Sum --> Out
