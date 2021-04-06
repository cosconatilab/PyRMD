# ![PyRMD-logo](https://user-images.githubusercontent.com/81375211/112626913-2c238880-8e31-11eb-8726-bfefcb5ebfa0.png)
PyRMD is a Ligand-Based Virtual Screening tool written in Python. The project is being developed by the [Cosconati Lab](https://sites.google.com/site/thecosconatilab/home) from [University of Campania Luigi Vanvitelli](https://international.unicampania.it/index.php/en/). 
Supported by an [AIRC](https://www.airc.it/) Fellowship for Italy.

Manuscript in preparation.

Authors: [Dr. Giorgio Amendola](mailto:giorgio.amendola@unicampania.it) and [Prof. Sandro Cosconati](mailto:sandro.cosconati@unicampania.it)

# Installation
First, users should download and install [Anaconda](https://www.anaconda.com/products/individual).

Once Anaconda has been installed, download the files from this repository and from the terminal (Linux, MacOs) or the Command Prompt (Windows) enter: 
```bash
create -f pyrmd_environment.yml
```
Follow the instructions appearing on the terminal until the environment installation is complete.

Adjust the `configuration_file.ini` with a text editor according to your preferences.To use PyRMD, activate the `pyrmd` conda environment:
```bash
conda activate pyrmd
```
Then, you are ready to run the software:
```bash
python PyRMD_v_1.00.py configuration_file.ini
```

# Quick Guide
Coming soon

# RMD Algorithm
PyRMD implements the Random Matrix Discriminant (RMD) algorithm devised by [Lee et al.](https://www.pnas.org/content/116/9/3373) to identify small molecules endowed with biological activity. Parts of the RMD algorithm code were adapted from the [MATLAB version of the RMD](https://github.com/alphaleegroup/RandomMatrixDiscriminant) and a [Python implementation proposed by Laksh Aithani](https://towardsdatascience.com/random-matrix-theory-the-best-classifier-for-prediction-of-drug-binding-f82613fb48ed) of the [Random Matrix Theory](https://www.pnas.org/content/113/48/13564).
