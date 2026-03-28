# Metabolic Network Biomarker Benchmarking

A comprehensive benchmarking framework for metabolic network-based biomarker discovery integrating transcriptomics, metabolomics, and genome-scale metabolic models (GEMs).

This repository enables multi-omics integration, constraint-based modeling, and evaluation of predictive performance, network rewiring, and biomarker stability.

## Repository Structure & Workflow

The codebase is organized into five primary analytical modules.

## 1. Mapping & Model Resources

This module contains metabolic model files and mapping utilities used across all analyses.

- `Recon3D.json`  
  Genome-scale metabolic model used as the core GEM framework.

- `new-synonym-mapping.json`  
  Metabolite synonym and standardization mappings.

- `mapping.py`  
  Functions for mapping metabolites and genes using GEM-based relationships.

- `Reverse_Metabolite_Mapper.py`  
  Reverse mapping pipeline to infer gene-level features from metabolite signatures.

## 2. Algorithm Analysis & Tier Analysis

This module evaluates metabolic network-based biomarker discovery algorithms and provides hierarchical biological validation.

- `Algorithm_Analysis.py`  
  Benchmarking TAMBOOR, TIMBR, and Modified TIMBR against each other and data-driven baselines.

- `tier_analysis.py`  
  Tier-based hierarchical evaluation of transcriptomic signals using GEM-derived insights.

## 3. Clinical Validation

This module assesses the clinical relevance and predictive performance of discovered biomarkers.

- `log2_fold_change.py`  
  Differential expression analysis using log2 fold change.

- `log2fc_vs_HR.py`  
  Association between log2 fold change and hazard ratios (HR), linking molecular signals to clinical outcomes.

## 4. Classification

This module contains methods for feature selection and supervised classification.

- `RFE_Classification.py`  
  Recursive Feature Elimination (RFE)-based classification for biomarker selection and validation.

## 5. Cross-Omics Analysis

This module investigates relationships between transcriptomic and metabolomic layers, focusing on integration, stability, and network structure.

### Integration Methods
- `SNF.py` — Similarity Network Fusion  
- `CCA.py` — Canonical Correlation Analysis  
- `multi_block_pls.py` — Multi-block Partial Least Squares  

### Stability & Module Discovery
- `stability.py`  
- `synergy.py`  
- `sparse_module_discovery.py`  

### Network & Topology Analysis
- `network_rewriting.py`  
- `topology.py`  
- `variance_decomposition.py`  

### Nonlinear Discovery & Distance Metrics
- `nonlinear_discovery.py`  
- `nonlinear_dependency.py`  
- `distance.py`  

### Visualization
- `heatmap_PCA.py`

## 🧠 Manuscript Status

The research associated with this repository is currently under review in a scientific journal.

Once the manuscript is published, this repository will be updated to include the full citation and publication details.

## 📂 Data Availability

The datasets used in this study are not publicly available due to access restrictions and are not included in this repository. The data sources are described in detail in the manuscript.

All analysis code is fully provided to support methodological reproducibility.

## How to Use

Each script in this repository is designed to be executed independently.

### General Usage

```bash
python script_name.py
