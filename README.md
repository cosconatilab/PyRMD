# ![PyRMD-logo](https://user-images.githubusercontent.com/81375211/112626913-2c238880-8e31-11eb-8726-bfefcb5ebfa0.png)
PyRMD is a Ligand-Based Virtual Screening tool written in Python powered by machine learning. The project is being developed by the [Cosconati Lab](https://sites.google.com/site/thecosconatilab/home) from [University of Campania Luigi Vanvitelli](https://international.unicampania.it/index.php/en/). 
Supported by the [AIRC](https://www.airc.it/) Fellowship for Italy Clementina Colombatti.

Please check our open-access manuscript published in the Journal of Chemical Information and Modeling for the full explanation of PyRMD functions
https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00653

Authors: [Dr. Giorgio Amendola](mailto:giorgio.amendola@unicampania.it) and [Prof. Sandro Cosconati](mailto:sandro.cosconati@unicampania.it)

# Installation and Usage
First, users should download and install [Anaconda](https://www.anaconda.com/products/individual).

Once Anaconda has been installed, download the files from this repository and from the terminal (Linux, MacOs) or the Command Prompt (Windows) enter: 
```bash
create -f pyrmd_environment.yml
```
Follow the instructions appearing on the terminal until the environment installation is complete.

Adjust the `configuration_file.ini` with a text editor according to your preferences. To use PyRMD, activate the `pyrmd` conda environment:
```bash
conda activate pyrmd
```
Then, you are ready to run the software:
```bash
python PyRMD_v1.02.py configuration_file.ini
```

If you need a clean configuration file, running PyRMD without any argument, like this:
```bash
python PyRMD_v1.02.py 
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
python PyRMD_v1.02.py configuration_benchmark.ini
```
At the end of the calculations, the `benchmark_results.csv` file will include the averaged benchmark metrics (TPR, FPR, Precision, F-Score, ROC AUC, PRC AUC, and BEDROC) across all the folds and repetitions. Also, the plots `ROC_curve.png` and `PRC_curve.png` will be generated.
## Screening
The screening test case trains PyRMD with the MET ChEMBL bioactivity data (the same used in the benchmark) and proceeds to screen a small sample of randomly extracted compounds from [MCULE](https://mcule.com/). These settings are specified in the `configuration_screening.ini` file that can be easily modified. 
To activate the conda environment and run the screening, enter:
```bash
conda activate pyrmd
python PyRMD_v1.02.py configuration_screening.ini
```
At the end of the calculations, the `database_predictions.csv` file will report a summary of the molecules predicted to be active against MET. For each compound, the file will include the molecule SMILES string, the RMD confidence score(the higher the better), the most similar training active compound and its relative similarity, and a flag indicating if it is a potential PAINS. Also, the `predicted_actives.smi` SMILES file will be created to be readily used with other cheminformatics/molecular modeling software.

# Optimizing PyRMD Performance
Poor performance may be the result of several factors, which include:
 - Insufficient training data, either for the active set or the inactive one
 - Many more inactive compounds than actives in the training set
 - Training data sets which comprises compounds with a different mechanism of action
 - Inadequate epsilon cutoff values

The above cases and others are discussed in the [JCIM article](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00653). Importantly, adjusting the epsilon cutoff values may readily improve poor performance, especially with regards to TPR, FPR, Precision, and F-Score, as they impact the classification thresholds. The default cutoff values for both the active training set and the inactive training set is 0.95, so that 95% of the training set will be considered in the model building step. This allows to account for some tolerance to the presence/absence of chemical motifs. Higher values (e.g. closer to 1) for the active training cutoff generally result in more true positives and false positives alike. Instead, a higher cutoff for the inactive set fitting should mainly decrease the false positive rate with some effect on the TPR. 

We suggest benchmarking using the possible combinations of the following epsilon cutoff values: 0.84–0.95–0.98 for epsilon_active and 0.7–0.84–0.95–0.98 for epsilon_cutoff_inactive and identify the combination with the best TPR/FPR tradeoff. Further information about the epsilon cutoff values are available in the [JCIM article](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00653) and in its [Supporting Information](https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.1c00653/suppl_file/ci1c00653_si_001.pdf). 

# RMD Algorithm
PyRMD implements the Random Matrix Discriminant (RMD) algorithm devised by [Lee et al.](https://www.pnas.org/content/116/9/3373) to identify small molecules endowed with biological activity. Parts of the RMD algorithm code were adapted from the [MATLAB version of the RMD](https://github.com/alphaleegroup/RandomMatrixDiscriminant) and a [Python implementation proposed by Laksh Aithani](https://towardsdatascience.com/random-matrix-theory-the-best-classifier-for-prediction-of-drug-binding-f82613fb48ed) of the [Random Matrix Theory](https://www.pnas.org/content/113/48/13564).
