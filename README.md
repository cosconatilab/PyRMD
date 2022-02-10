# ![PyRMD-logo](https://user-images.githubusercontent.com/81375211/112626913-2c238880-8e31-11eb-8726-bfefcb5ebfa0.png)
PyRMD is a Ligand-Based Virtual Screening tool written in Python powered by machine learning. The project is being developed by the [Cosconati Lab](https://sites.google.com/site/thecosconatilab/home) from [University of Campania Luigi Vanvitelli](https://international.unicampania.it/index.php/en/). 
Supported by the [AIRC](https://www.airc.it/) Fellowship for Italy Clementina Colombatti.

Please check our open-access manuscript published in the Journal of Chemical Information and Modeling for the full explanation of PyRMD functions
https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00653

A new protocol, **PyRMD2DOCK**, has been devised as an approach to approximate the score of molecular docking experiments using PyRMD. An outline of the protocol is described in the **PyRMD2DOCK** section at the bottom of this document.

Authors: [Dr. Giorgio Amendola](mailto:giorgio.amendola@unicampania.it) and [Prof. Sandro Cosconati](mailto:sandro.cosconati@unicampania.it)

# Installation and Usage
First, users should download and install either Anaconda or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

Once conda has been installed, download the files from this repository and from the terminal (Linux, MacOs) or the Command Prompt (Windows) enter: 
```bash
conda env create -f pyrmd_environment.yml
```
This will install to install the `pyrmd` conda environment. Follow the instructions appearing on the terminal until the environment installation is complete.

Adjust the `configuration_file.ini` with a text editor according to your preferences. To use PyRMD, activate the `pyrmd` conda environment:
```bash
conda activate pyrmd
```
Then, you are ready to run the software:
```bash
python PyRMD_v1.03.py configuration_file.ini
```

If you need a clean configuration file, running PyRMD without any argument, like this:
```bash
python PyRMD_v1.03.py 
```
It will automatically generate a `default_config.ini` with default settings.

If you are having troubles in correctly setting up PyRMD or optimizing its performance, [get in touch with us](mailto:giorgio.amendola@unicampania.it).
# Tutorials
In the `tutorials` folder are present two test cases, one for the benchmark mode and another for the screening mode, with all the files and the configurations already set up. Users only need to run PyRMD in the respective folders. 
## Benchmark
The benchmark test case allows to benchmark PyRMD performance using the target bioactivity data downloaded from [ChEMBL](https://www.ebi.ac.uk/chembl/target_report_card/CHEMBL3717/) for the tyrosine-kinase MET. The benchmark employs a Repeats Stratified K-Fold approach with 5 folds and 3 repetions. MET decoy compounds downloaded from the [DUD-E](http://dude.docking.org/) are also included in the folder to be used as an additional test set. These settings are specified in the `configuration_benchmark.ini` file that can be easily modified.
To activate the conda environment and run the benchmark, enter:
```bash
conda activate pyrmd
python PyRMD_v1.03.py configuration_benchmark.ini
```
At the end of the calculations, the `benchmark_results.csv` file will include the averaged benchmark metrics (TPR, FPR, Precision, F-Score, ROC AUC, PRC AUC, and BEDROC) across all the folds and repetitions. Also, the plots `ROC_curve.png` and `PRC_curve.png` will be generated.
## Screening
The screening test case trains PyRMD with the MET ChEMBL bioactivity data (the same used in the benchmark) and proceeds to screen a small sample of randomly extracted compounds from [MCULE](https://mcule.com/). These settings are specified in the `configuration_screening.ini` file that can be easily modified. 
To activate the conda environment and run the screening, enter:
```bash
conda activate pyrmd
python PyRMD_v1.03.py configuration_screening.ini
```
At the end of the calculations, the `database_predictions.csv` file will report a summary of the molecules predicted to be active against MET. For each compound, the file will include the molecule SMILES string, the RMD confidence score(the higher the better), the most similar training active compound and its relative similarity, and a flag indicating if it is a potential PAINS. Also, the `predicted_actives.smi` SMILES file will be created to be readily used with other cheminformatics/molecular modeling software.

# Training Data
Please check the **"Training Dataset Preparation"** and **"Benchmarking  Training  Data"** sections in the [Supporting Information PDF of the JCIM article](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00653) for a description of what kind of training data to use with PyRMD

# Optimizing PyRMD Performance
Poor performance may be the result of several factors, which include:
 - Insufficient training data, either for the active set or the inactive one
 - Many more inactive compounds than actives in the training set
 - Training data sets which comprises compounds with a different mechanism of action
 - **Inadequate epsilon cutoff values**

The above cases and others are discussed in the [JCIM article](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00653). Importantly, adjusting the epsilon cutoff values may readily improve poor performance, especially with regards to TPR, FPR, Precision, and F-Score, as they impact the classification thresholds. The default cutoff values for both the active training set and the inactive training set is 0.95, so that 95% of the training set will be considered in the model building step. This allows to account for some tolerance to the presence/absence of chemical motifs. Higher values (e.g. closer to 1) for the active training cutoff generally result in more true positives and false positives alike. Instead, a higher cutoff for the inactive set fitting should mainly decrease the false positive rate with some effect on the TPR. 

As a possible starting point, we suggest benchmarking using the possible combinations of **the following epsilon cutoff values: 0.84–0.95–0.98 for epsilon_active and 0.7–0.84–0.95–0.98 for epsilon_cutoff_inactive** and identify the combination with the best TPR/FPR tradeoff. Further information about the epsilon cutoff values are available in the [JCIM article](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00653) and in its Supporting Information PDF. 

# RMD Algorithm
PyRMD implements the Random Matrix Discriminant (RMD) algorithm devised by [Lee et al.](https://www.pnas.org/content/116/9/3373) to identify small molecules endowed with biological activity. Parts of the RMD algorithm code were adapted from the [MATLAB version of the RMD](https://github.com/alphaleegroup/RandomMatrixDiscriminant) and a [Python implementation proposed by Laksh Aithani](https://towardsdatascience.com/random-matrix-theory-the-best-classifier-for-prediction-of-drug-binding-f82613fb48ed) of the [Random Matrix Theory](https://www.pnas.org/content/113/48/13564).

# Updates

Version 1.03 fixes a rare bug that led to crashes when processing more exotic compounds in screening mode.

# PyRMD2DOCK - A New Protocol
Even though PyRMD was conceived to be used with experimentally validated data, very recent tests confirmed that it can also be used to approximate the score of docking experiments. In our tests, we used [AutoDock-GPU](https://github.com/ccsb-scripps/AutoDock-GPU) to dock on a protein target approximately one million compounds randomly extracted from the [ZINC](https://zinc20.docking.org/substances/home/) tranche of in stock drug-like compounds (~10 million compounds).

The docking energies (lowest energy clusters only) were extracted and plotted. We selected the compounds whose lowest energy docking score was =< -9.5 as active training set, for a total of ~8000 compounds. While the molecules with a lowest energy docking score >= -4.5 were chosen as inactive training set, for a total of ~3000 compounds. In our experience, having a 2:1 or 3:1 ratio of actives/inactives in the training set is favorable for PyRMD performance. Thus, benchmarking experiments to find the best performing epsilon cutoff values combinations were run, using ~25000 compounds with a docking score between -4.5 and -5.5 as decoys.
The combination of epislon cutoff values which gave the best F-score (0.90) and high ROC AUC (0.99) was selected. 

Using PyRMD, these settings were then used to screen the ZINC compounds not docked initially (~9 million compounds). The best 10000 molecules according to the PyRMD RMD score were then docked using AutoDock-GPU on the protein target. Notably, the distribution of the docking scores showed a marked increase towards lower energy scores. The average docking score of the original docking experiment on one million compounds was of -7.5. While the 10000 molecules selected using PyRMD averaged a docking score of -9.3, demonstrating that **PyRMD is highly capable of identifying the compounds which score better according to the AutoDock scoring function in a short amount of time**. 

Potentially, this approach can be used on any kind of target on which docking can be performed, and with any kind of docking software, to screen ultra-large commercial libraries of even billions of compounds much faster than docking softwares. Compared to using PyRMD with experimentally validated data, **the PyRMD2DOCK approach** allows to identify even more diverse chemical scaffolds, **by combining the speed of 2D-based calculations with the ability of structure-based virtual screening to pick novel chemotypes**.

For more information on the PyRMD2DOCK protocol, [get in touch with us](mailto:giorgio.amendola@unicampania.it).

