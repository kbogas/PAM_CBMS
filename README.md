# PAM_CBMS
Analysing Biomedical Knowledge Graphs usingPrime Adjacency Matrices

Accompanying code for CBMS_2023 submission.

The requirements for this project can be installed by running:

```python
pip -r requirements.txt
````

Python version used: Python 3.9.16

Data are expected to be downloaded in a folder in "./data/" (here, on the top-level folder)

The datasets used in must be downloaded from the corresponding (open) sources:
- [DDB14](https://github.com/hwwang55/PathCon/tree/master/data/DDB14) used in the scalability experiment.
- [PharmKG](https://zenodo.org/record/4077338) used in the scalability experiment.
- [HetioNet](https://github.com/hetio/hetionet) used in the scalability and the drug-repurposing experiment.
- [DRKG](https://github.com/gnn4dr/DRKG) used in the scalability and the drug-repurposing experiment.
- [PrimeKG](https://github.com/mims-harvard/PrimeKG) used in the scalability experiment.


The *utils.py* and *pam_creation.py* files, contain functionality code to support the proposed framework and facilitate the experiments.

The rest of the files are used to reproduce the results present in the article:
- The *scalability_test_CBMS.py* reproduces the usability results presented in Section 3.1.
- The *DRKG_Drug_Repurposing.ipynb* reproduces the Drug-Repurposing case study in Section 3.2.
- The *Hetionet_Metapaths_Extraction.ipynb* reproduces the Metapath Extraction case study in Section 3.3.

