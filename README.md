# Metabolic Biomarker Discovery Benchmarking Framework

A benchmarking framework for evaluating constraint-based genome-scale metabolic modeling approaches (TAMBOOR, TIMBR, E-TIMBR) using matched transcriptomic and metabolomic data across Alzheimer’s disease, breast cancer, and colon cancer cohorts.

The repository focuses on assessing the consistency, predictive performance, and stability of inferred metabolic signatures derived from Recon3D-based models compared to experimentally measured metabolomic profiles.

## Mapping & Model Resources

This module contains metabolic model files and mapping utilities used across all analyses.

- `Recon3D.json`  
  Genome-scale metabolic model used as the core GEM framework.

- `new-synonym-mapping.json`  
  Metabolite synonym and standardization mappings.

- `mapping.py`  
  Functions for mapping metabolites and genes using GEM-based relationships.

- `Reverse_Metabolite_Mapper.py`  
  Reverse mapping pipeline to infer gene-level features from metabolite signatures.

## Analysis Codes

- `Algorithm_Analysis.py`  
  
- `RFE_Recon3D.py`  
  
- `PCA_analysis.py`  

- `corelation.py`  

- `discrimination_analysis.py`  

- `distance_analysis.py`  

- `entropy.py`  

- `signal_analysis.py`  

- `tier_analysis.py`  

- `tier_analysis_2.py`

- `recon3D_performance_analysis.py`  

## 🧠 Manuscript Status

The research associated with this repository is currently under review in a scientific journal.

Once the manuscript is published, this repository will be updated to include the full citation and publication details.

## 📂 Data Availability

The datasets used in this study are not publicly available due to access restrictions and are not included in this repository. The data sources are described in detail in the manuscript.

All analysis code is fully provided to support methodological reproducibility.

## How to Use

The workflow is partially sequential. The core pipeline should be executed in a defined order, followed by independent downstream analyses.

### Recommended Execution Order

1. **Mapping & preprocessing stage**
   - `mapping.py`
   - `Reverse_Metabolite_Mapper.py`

2. **Core benchmarking baseline**
   - `Algorithm_Analysis.py`

3. **Reference-based feature selection**
   - `RFE_Recon3D.py`

### After completing the steps above, the remaining analysis scripts can be executed independently in any order:

- `PCA_analysis.py`  
- `corelation.py`  
- `discrimination_analysis.py`  
- `distance_analysis.py`  
- `entropy.py`  
- `signal_analysis.py`  
- `tier_analysis.py`  
- `tier_analysis_2.py`  
- `recon3D_performance_analysis.py`

---

### General Usage

```bash
python script_name.py
