# Update to models developed for manuscript titled: _Artificial Intelligence Outperforms a Nomogram for Osteoradionecrosis Prognostication Following Fibula Free Flap Reconstruction in Oral Cancer Patients_

<a href="https://doi.org/10.5281/zenodo.22683215"><img src="https://zenodo.org/badge/1359569761.svg" alt="DOI"></a>

## Description
This project builds on the implementation of various Machine Learning (ML) models and a logistic regression-based nomogram to predict post-operative occurrence of osteoradionecrosis (ORN) in head and neck cancer patients receiving fibula flap reconstruction after tumor excision and segmental mandibulectomy. The code for the original manuscript<sup>1</sup> can be found at 
https://github.com/AnthonyMatarr/ML-ORN-Pred.


<sup>1</sup>Matar DY, Mackert GA, Matar AY, Chen AC, Panayi AC, Knoedler L, Knoedler S, Yang R, Mady LJ, Kao HK. Artificial intelligence outperforms a nomogram for osteoradionecrosis prognostication following fibula free flap reconstruction in oral cancer patients. J Stomatol Oral Maxillofac Surg. 2026 Feb;127(1):102584. doi: 10.1016/j.jormas.2025.102584. Epub 2025 Oct 1. PMID: 41043770.

## Project layout
- Notebooks/: end-to-end machine learning workflows (data cleaning, preprocessing, tuning, evaluation, and feature importance derivation)

- src/: reusable functions shared across notebooks, including a global seed/random state

- data/: local data storage (not tracked; raw data not available for public use at this time)
 
- models/traind: trained models available for use

## License: MIT
- Code licensed under MIT
- No data are included

## Usage
Note that models are publicly available. If used or validated elsewhere, please refer to this codebase with citation:

Matar, A., & Matar, D. (2026). AnthonyMatarr/ML-ORN-Pred-update: v2.0.0 - Updated development of ORN prediction models (Version v2.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.22683216


