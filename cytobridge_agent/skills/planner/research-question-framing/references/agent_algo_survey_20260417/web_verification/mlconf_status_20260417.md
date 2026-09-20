# ML Conference Publication Verification

Date: 2026-04-17

## Verified as formally published

### Flow Matching on General Geometries
- Local PDF title: `Flow Matching on General Geometries.pdf`
- Verified status: formally published at ICLR 2024
- Primary evidence:
  - ICLR Proceedings page states `International Conference on Learning Representations 2024 (ICLR 2024) Conference`
  - URL: `https://proceedings.iclr.cc/paper_files/paper/2024/hash/d1f9936d3be6997ffffab692977eebe6-Abstract-Conference.html`

### Metric Flow Matching for Smooth Interpolations on the Data Manifold
- Local PDF title: `Metric Flow Matching for Smooth Interpolations on the Data Manifold.pdf`
- Verified status: formally published at NeurIPS 2024
- Primary evidence:
  - NeurIPS virtual poster page lists the paper as `2024 Poster` with proceedings link
  - URL: `https://nips.cc/virtual/2024/poster/94221`

### Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport
- Local PDF title: `Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport.pdf`
- Verified status: formally published at ICLR 2025
- Primary evidence:
  - ICLR proceedings abstract page explicitly lists it as part of `ICLR 2025`
  - URL: `https://proceedings.iclr.cc/paper_files/paper/2025/hash/32b8a612105de5c22db337b774ce7b61-Abstract-Conference.html`
  - Supporting evidence: author site and dblp both label it `ICLR 2025`

### Modeling Complex System Dynamics with Flow Matching Across Time and Conditions
- Local PDF title: `Modeling Complex System Dynamics with Flow Matching Across Time and Conditions.pdf`
- Verified status: formally published at ICLR 2025
- Primary evidence:
  - ICLR proceedings abstract page explicitly lists it as part of `ICLR 2025`
  - URL: `https://proceedings.iclr.cc/paper_files/paper/2025/hash/206b764b00195b8c896c5af8bde80726-Abstract-Conference.html`

### TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics
- Local PDF title: `TrajectoryNet A Dynamic Optimal Transport Network for Modeling Cellular Dynamics.pdf`
- Verified status: formally published at ICML 2020
- Why this matters:
  - The local registry heuristic can misread it because the stored venue string is malformed and contains arXiv / PubMed-style fragments
  - It should be normalized as a conference paper, not a journal/preprint-only record
- Primary evidence:
  - PMLR proceedings page states `Proceedings of the 37th International Conference on Machine Learning`
  - URL: `https://proceedings.mlr.press/v119/tong20a.html`

### Variational Mixtures of ODEs for Inferring Cellular Gene Expression Dynamics
- Local PDF title: `Variational Mixtures of ODEs for Inferring Cellular Gene Expression Dynamics.pdf`
- Verified status: formally published at ICML 2022
- Why this matters:
  - The local registry heuristic can leave it outside the ML conference bucket because the APA string in the KB is preprint-like
  - It should be normalized as an ICML paper if ultimately included
- Primary evidence:
  - PMLR proceedings page states `Proceedings of the 39th International Conference on Machine Learning`
  - URL: `https://proceedings.mlr.press/v162/gu22a.html`

## Not formally published yet based on current evidence

### ContextFlow: Context-Aware Flow Matching For Trajectory Inference From Spatial Omics Data
- Local PDF title: `ContextFlow Context-Aware Flow Matching for Trajectory Inference from Spatial Omics Data.pdf`
- Current status: still appears to be an arXiv preprint / ICLR 2026 submission rather than a formally published ICLR paper
- Primary evidence:
  - arXiv record shows only preprint status and arXiv DOI
  - URL: `https://arxiv.org/abs/2510.02952`
  - Search results show an `Under review as a conference paper at ICLR 2026` PDF rather than an accepted proceedings page
- Consequence for inclusion:
  - It does **not** satisfy the strict “formally发表于 NeurIPS / ICML / ICLR” criterion at the time of this audit
  - It may still be discussed later in synthesis as a recent supporting preprint if needed, but not as a formal ML conference inclusion

### CellFlow enables generative single-cell phenotype modeling with flow matching
- Local PDF title: `CellFlow enables generative single-cell phenotype modeling with flow matching.pdf`
- Current status: still appears to be a bioRxiv preprint rather than a formally published top-conference paper
- Primary evidence:
  - J-GLOBAL record explicitly labels the item as `bioRxiv` and `Document type: Preprint`
  - URL: `https://jglobal.jst.go.jp/en/public/202502210858931070`
- Consequence for inclusion:
  - It should not be counted as a formal ML-conference inclusion unless separate later evidence shows an accepted NeurIPS / ICML / ICLR version
