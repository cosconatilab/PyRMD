# ![PyRMD-logo](https://user-images.githubusercontent.com/81375211/112626913-2c238880-8e31-11eb-8726-bfefcb5ebfa0.png)
PyRMD is a Ligand-Based Virtual Screening tool written in Python powered by machine learning. The project is being developed by the [Cosconati Lab](https://sites.google.com/site/thecosconatilab/home) from [University of Campania Luigi Vanvitelli](https://international.unicampania.it/index.php/en/). 
Supported by the [AIRC](https://www.airc.it/) Fellowship for Italy Clementina Colombatti.

Please check our open-access manuscript published in the Journal of Chemical Information and Modeling for the full explanation of PyRMD functions
https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00653

A new protocol, **PyRMD-2-DOCK**, has been devised as an approach to approximate the score of molecular docking experiments using PyRMD. An outline of the protocol is described in the **PyRMD-2-DOCK** section at the bottom of this document.

Authors: Dr. Giorgio Amendola and [Prof. Sandro Cosconati](mailto:sandro.cosconati@unicampania.it)

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

If you are having troubles in correctly setting up PyRMD or optimizing its performance, [get in touch with us](mailto:sandro.cosconati@unicampania.it).
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

## Training Data
Please check the **"Training Dataset Preparation"** and **"Benchmarking  Training  Data"** sections in the [Supporting Information PDF of the JCIM article](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00653) for a description of what kind of training data to use with PyRMD

## Optimizing PyRMD Performance
Poor performance may be the result of several factors, which include:
 - Insufficient training data, either for the active set or the inactive one
 - Many more inactive compounds than actives in the training set
 - Training data sets which comprises compounds with a different mechanism of action
 - **Inadequate epsilon cutoff values**

The above cases and others are discussed in the [JCIM article](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00653). Importantly, adjusting the epsilon cutoff values may readily improve poor performance, especially with regards to TPR, FPR, Precision, and F-Score, as they impact the classification thresholds. 

The default cutoff values for both the active training set and the inactive training set is 0.95, so that 95% of the training set will be considered in the model building step. This allows to account for some tolerance to the presence/absence of chemical motifs. Higher values (e.g. closer to 1) for the active training cutoff generally result in more true positives and false positives alike. Instead, a higher cutoff for the inactive set fitting should mainly decrease the false positive rate with some effect on the TPR. 

As a possible starting point, we suggest benchmarking using the possible combinations of **the following epsilon cutoff values: 0.84–0.95–0.98 for epsilon_active and 0.7–0.84–0.95–0.98 for epsilon_cutoff_inactive** and identify the combination with the best TPR/FPR tradeoff. If the suggested values combinations yield dissatisfactory results, a more thorough sampling of the epsilon cutoff values could be necessary by including greater ranges and more granular values.

For instance, in the most challenging cases, we tested all the inactive cutoff values from 0.01 to 0.99 with a step of 0.03, and all the active cutoff values from 0.80 to 0.99 with a step of 0.03. Indeed, Bash scripting can be used to automate the sampling of the different cutoff combinations. 

Further information about the epsilon cutoff values are available in the [JCIM article](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00653) and in its Supporting Information PDF. 

## RMD Algorithm
PyRMD implements the Random Matrix Discriminant (RMD) algorithm devised by [Lee et al.](https://www.pnas.org/content/116/9/3373) to identify small molecules endowed with biological activity. Parts of the RMD algorithm code were adapted from the [MATLAB version of the RMD](https://github.com/alphaleegroup/RandomMatrixDiscriminant) and a [Python implementation proposed by Laksh Aithani](https://towardsdatascience.com/random-matrix-theory-the-best-classifier-for-prediction-of-drug-binding-f82613fb48ed) of the [Random Matrix Theory](https://www.pnas.org/content/113/48/13564).

# PyRMD2Dock - A New Protocol
New studies highlight the significance of employing exceptionally large molecular databases in Structure-Based Virtual Screening (SBVS) campaigns, as it significantly enhances the probability of uncovering promising and innovative drug candidates. To tackle this issue, we have introduced **PyRMD2Dock**, an innovative computational approach that combines the rapidity of our AI-driven Ligand-Based Virtual Screening (LBVS) tool, PyRMD, with the widely utilized docking software [AutoDock-GPU](https://github.com/ccsb-scripps/AutoDock-GPU). This approach involves docking a small subset of the target database, which serves as training data to construct a Machine Learning (ML) model capable of predicting the docking performance of an extensive collection of chemical compounds.

![image](https://github.com/cosconatilab/PyRMD/assets/131974589/6106b42b-4595-4fc9-9b7e-8623ea5d4510)

The PyRMD2Dock approach integrates the inherent benefits of Structure-Based Virtual Screening (SBVS) with the capabilities of AI-driven Ligand-Based Virtual Screening (LBVS), enabling the efficient screening of billions of ligands within a manageable timeframe and with limited computational resources.

In our tests, we used AutoDock-GPU to dock, on five protein targets, approximately one million compounds randomly extracted from the [ZINC](https://zinc20.docking.org/substances/home/) tranche of in stock drug-like compounds (~10 million compounds), to be used as training set.  

To validate the predictive capabilities of PyRMD2Dock, we conducted benchmarking experiments and performed post hoc docking of compounds selected by the protocol. The obtained results demonstrate that the models developed for five relevant pharmaceutical targets successfully identify compounds that exhibit favorable performance in subsequent docking calculations.

## PyRMD2Dock - Implementation

The PyRMD2Dock approach provides a user-friendly experience through its simple workflow and automated process. However, certain aspects of the method require user involvement. To assist with these user-dependent tasks, we have developed a set of scripts that can be executed effortlessly with a single-line command in the terminal.

### Renaming Column Names

To successfully execute the entire PyRMD2Dock protocol, it is necessary to modify the column names associated with the ligand name and corresponding SMILES representation within the initial database used for the studies. We recommend renaming the column containing the ligand name as _"Ligand"_ and the column containing the SMILES representation as _"smiles"_. This adjustment ensures smooth data processing and adherence to the expected format throughout the protocol. Please note that this column renaming requirement applies specifically to the initial database used as input for the PyRMD2Dock protocol. Subsequent outputs and intermediate files generated during the protocol will automatically follow the standard naming conventions specified within the PyRMD2Dock implementation.

### Ligand Preparation

The initial training compounds need to be extracted from the database and converted into a suitable format for docking studies, such as a 3D file (.mol2 or .pdb) or any format required. It is crucial to prepare each ligand by enumerating the tautomers, protomers, and isomers, and starting with a low-energy ligand conformation. In our work, we employed "LigPrep" for ligand preparation. Refer to the docking methods section for more details on ligand preparation and docking calculations. After this step, the user must select the training set for docking calculations, resulting in the generation of a comma-separated .csv file containing docking results. To extract this information, we have implemented a python script named ["prepare_docking_ranking.py"](PyRMD2Dock_useful_scripts/).

### Benchmarking Calculations

For benchmarking calculations, active and inactive databases are required, with user-defined thresholds for each set. To accomplish this, execute the ["create_distribution_plot.py"](PyRMD2Dock_useful_scripts/) script using the ["prepare_docking_ranking.py"](PyRMD2Dock_useful_scripts/) resulting file to generate a distribution plot of the calculated binding free energy (ΔGAD4) values. Based on this plot, users can set their own thresholds for activity and inactivity. These thresholds are essential for constructing the activity and inactivity databases needed for benchmarking calculations. To generate these databases, use the ["get_actives_inactives.py"](PyRMD2Dock_useful_scripts/) script. For easy creation of input files with various user-defined settings, a script named ["prepare_benchmarking_dirs.py"](PyRMD2Dock_useful_scripts/) is provided. Once these preparations are complete, benchmarking calculations can be initiated.

### Model Selection and Screening

Following the benchmarking calculations, the user must select the best model based on the optimal tradeoff between true positive rate (TPR) and false positive rate (FPR). This selected model will be used for subsequent screening calculations.

To conduct screening in PyRMD2Dock, refer to the screening section of this page as reported above. After the screening calculations are completed, PyRMD generates a comma-separated .csv file containing various columns, including the confidence score (RMD_score). Using this score as a guide, the user needs to choose a subset of compounds for confirmative docking calculations. For ease of selection, we have developed a Python script named ["select_best_RMD.py"](PyRMD2Dock_useful_scripts/), which outputs two comma-separated .csv files: one containing the chosen subset along with all the information derived from PyRMD screening, and another containing a list of the selected compounds with their corresponding smiles strings. The latter file serves as a reference for subsequent docking calculations (ligand preparation and docking calculations).


For more information on the PyRMD-2-DOCK protocol, [get in touch with us](mailto:sandro.cosconati@unicampania.it).

## Updates

Version 1.03 fixes a rare bug that led to crashes when processing more exotic compounds in screening mode.
