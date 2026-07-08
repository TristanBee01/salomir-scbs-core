# salomir-scbs-core
## Overview
This repository contains the core architectural blueprints and agent configurations for the **Salomir SCBS**. Salomir was founded to solve a critical opacity problem in the AI cloud market: buyers procure billions in compute based on marketing claims (TFLOPS), but often realize 30-50% performance degradation due to hidden bottlenecks in network topology and storage I/O.

The SCBS utilizes "Artifact Parity"—deploying an immutable, locked Docker container across disparate cloud providers to isolate hardware faults from software variance.

## Methodology: Artifact Parity
Traditional benchmarks (e.g., MLPerf) allow providers to heavily tune compilers and drivers to pass the test. The SCBS prevents this. By freezing the CUDA environment, PyTorch version, and networking libraries inside the container, we guarantee the software environment is identical across all tests. 
*If the exact same Docker container runs 20% slower on Provider A vs. Provider B, it is mathematically proven to be an infrastructure fault.*

## Core Components
*   `scbs-agent/`: The Dockerfile and entry scripts for the immutable testing artifact.
*   `workloads/`: Synthetic Python scripts designed to trigger specific infrastructure bottlenecks (e.g., massive KV cache VRAM offloading to test PCIe/NVMe bandwidth).
*   `network-stress/`: Bash wrappers for NCCL `all_reduce` tests, designed to measure jitter and packet drop rates across InfiniBand and RoCEv2 fabrics during simulated Mixture of Experts (MoE) routing.

## The Goal
To move the cloud industry from static "point-in-time" benchmarks to continuous, autonomous, agent-driven infrastructure auditing.
