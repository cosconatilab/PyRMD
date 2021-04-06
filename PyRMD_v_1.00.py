#!/usr/bin/env python
# coding: utf-8

# PyRMD: AI-powered Virtual Screening
# Copyright (C) 2021  Dr. Giorgio Amendola, Prof. Sandro Cosconati
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# 
# If you used PyRMD in your work, or you require information regarding the software, please email the authors at giorgio.amendola@unicampania.it --- sandro.cosconati@unicampania.it
# 
# Please check our GitHub page for more information 
# <https://github.com/cosconatilab/PyRMD>
# 

# In[1]:


#Imports
import os
import re
import time
from pathlib import Path
import subprocess


import rdkit
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit import rdBase
from rdkit.Chem import rdMHFPFingerprint
from rdkit.ML.Scoring import Scoring
from rdkit.Chem.AtomPairs import Torsions 
from rdkit import DataStructs
from rdkit.Chem import FilterCatalog



import numpy as np, scipy.stats as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


import sklearn
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score
import statsmodels.stats.api as sms

import openbabel as ob
from openbabel import pybel


import warnings
warnings.filterwarnings('ignore')

from configparser import ConfigParser
import argparse
import sys
import gc


# In[2]:


default_config='''[MODE]

#Indicate if the program is to run in "benchmark", "screening", or "inactives" mode -- Default: benchmark
mode=benchmark

#Indicate the database file(s) to screen for the screening mode, otherwise leave the entry blank
db_to_screen =

#Specify the output file for the screening mode
screening_output = database_predictions.csv

#Indicate if in screening mode the active compounds should also be converted in a SDF file -- True or False -- Default: False 
sdf_results= False

# For benchmark mode, indicate the file where to output the benchmark results
benchmark_file = benchmark_results.csv

[TRAINING_DATASETS]

# Indicate if one or more ChEMBL databases will be used as training sets -- True or False -- Default: True
use_chembl = True

#Indicate the CHEMBL database(s) file path, otherwise leave the entry blank
chembl_file =

# Set to True if one or more non-CHEMBL databases of active compounds are to be used for training the algorithm, otherwise set to False -- Default: False
use_actives = False

#Indicate the non-CHEMBL active compounds database(s) (SMILES file) path, otherwise leave the entry blank
actives_file =

# Set to True if one or more non-CHEMBL databases(s) of inactive compounds are to be used for training the algorithm, otherwise set to False
use_inactives = False

#Indicate the non-CHEMBL inactive compounds database(s) (SMILES file) file path, otherwise leave the entry blank
inactives_file =


[FINGERPRINTS]

#Fingerprint type - supported formats: rdkit, tt, mhfp, avalon, and ecfp -- Default: mhfp
#For fcfp set fp_type = ecfp and features = True
fp_type = mhfp

# lenght of the fingerprint - typical lenghts are 1024, 2048, and 4096 - longer fingerprints require more memory and are slower to process -- Default: 2048  
nbits = 2048

# Include explicit hydrogens in the fingerprint: True or False -- Default: True
explicit_hydrogens = True

#ecfp/mhfp specific parameters
iterations = 3
chirality = False

#ecfp specific parameters
redundancy = True
features = False

[DECOYS]

#Set to True if external decoys are to be used for benchmarking the algorithm, otherwise set to False -- Default: False
use_decoys = False

#In case external decoy compounds are to be used for the benchmark, indicate the decoy database file path, otherwise leave the entry blank
decoys_file =

# Particularly large decoy databases may severely slow down the benchmark process, setting a sample number of decoys to employ can help speed it up -- Default: 1000000
sample_number = 1000000

[CHEMBL_THRESHOLDS]

# Compounds will be considered active if they are reported to have a value of IC50, EC50, Ki, Kd, or potency, inferior to the "activity_threshold" expressed in nM. They will be classified as inactive if their IC50, EC50, Ki, Kd, or potency will be greater than the "inactivity_threshold" expressed in nM, or if their inhibition rate is lower than the "inhibition_threshold" rate -- Default values: activity_threshold = 1001; inactivity_threshold = 39999; inhibition_threshold = 11
activity_threshold = 1001
inactivity_threshold = 39999
inhibition_threshold = 11

[KFOLD_PARAMETERS]

#For statical benchmarking purposes, indicate the number of splits for the benchmark mode -- Default: 5
n_splits = 5

#For statical benchmarking purposes, indicate how many times the benchmarking calculation should be run -- Default: 3
n_repeats = 3

[TRAINING_PARAMETERS]

# Cutoff values that set the percentile of the distance between the compounds in the training set and their projections in the training linear subspace. The resulting maximum projection distance, the epsilon parameter, will be used to classify unknown compounds. Insert a float ranging from 0 to 1 -- Default: 0.95
epsilon_cutoff_actives=0.95
epsilon_cutoff_inactives=0.95

[STAT_PARAMETERS]

# F-Score beta value. Beta > 1 gives more weight to TPR, while beta < 1 favors precision. 
beta= 1

# Bedroc alpha value. 
alpha = 0.20


[FILTER]

#Filter from the SCREENED DATABASE compounds that are not within the specified ranges set below: True or False
filter_properties = False

# If the properties of a compound are not within the specified ranges (specified as integers), the compound will be discarded

molwt_min = 200
logp_min = -5
hdonors_min = 0
haccept_min =0
rotabonds_min =0
heavat_min = 15

molwt_max = 600
logp_max = 5
hdonors_max = 6
haccept_max = 11
rotabonds_max = 9
heavat_max = 51


'''
p = Path('temp_conf_8761.ini')
p.write_text(default_config)


# In[3]:


ap = argparse.ArgumentParser()
argv = sys.argv[1:]
if len(argv) ==0:
    p = Path('default_config.ini')
    p.write_text(default_config)
    print('Configuration file path missing, a default_config.ini file has been generated to use as template')
    os.remove('temp_conf_8761.ini')
    sys.exit()
ap.add_argument('config', nargs=1, help='configuration file path')
config_file = vars(ap.parse_args())['config']
 

# In[4]:


def string_or_list(prog_input):
    input_list=prog_input.split()
    if len(input_list) ==1:
        return input_list[0]
    elif len(input_list) >1:
        return input_list
    else:
        return prog_input


# In[5]:



config = ConfigParser()
config.read('temp_conf_8761.ini')
config.read(config_file)
default_keys = ['default','standard','']
os.remove('temp_conf_8761.ini')

#MODE
mode=config.get('MODE','mode')
verbose=False
score=True
sdf_results=config.getboolean('MODE','sdf_results')
db_to_screen= string_or_list(config.get('MODE','db_to_screen'))
benchmark_file= config.get('MODE','benchmark_file')
gray = False
inactives_similarity = False
inactives_similarity_file = None
screening_output = config.get('MODE','screening_output')
temporal = False
temporal_file = None

#TRAINING DATASETS
use_chembl=config.getboolean('TRAINING_DATASETS','use_chembl')
chembl_file = string_or_list(config.get('TRAINING_DATASETS','chembl_file'))

use_external_actives=config.getboolean('TRAINING_DATASETS','use_actives')
use_external_inactives= config.getboolean('TRAINING_DATASETS','use_inactives')   
actives_file= string_or_list(config.get('TRAINING_DATASETS','actives_file'))
inactives_file= string_or_list(config.get('TRAINING_DATASETS','inactives_file'))

#Fingerprints Parameters
fp_type = config.get('FINGERPRINTS','fp_type')
explicit_hydrogens = config.getboolean('FINGERPRINTS','explicit_hydrogens')
iterations = config.getint('FINGERPRINTS','iterations')
nbits = config.getint('FINGERPRINTS','nbits')
chirality = config.getboolean('FINGERPRINTS','chirality')
redundancy = config.getboolean('FINGERPRINTS','redundancy')
features = config.getboolean('FINGERPRINTS','features')

#DECOYS
use_external_decoys=config.getboolean('DECOYS','use_decoys')
decoys_file = string_or_list(config.get('DECOYS','decoys_file'))
sample_number=config.getint('DECOYS','sample_number')

#CHEMBL THRESHOLDS
activity_threshold = config.getint('CHEMBL_THRESHOLDS','activity_threshold')
inactivity_threshold = config.getint('CHEMBL_THRESHOLDS','inactivity_threshold')
inhibition_threshold = config.getint('CHEMBL_THRESHOLDS','inhibition_threshold')

#KFOLD PARAMETERS
n_splits = config.getint('KFOLD_PARAMETERS','n_splits')
n_repeats = config.getint('KFOLD_PARAMETERS','n_repeats')

#TRAINING PARAMETERS
threshold = config.getfloat('TRAINING_PARAMETERS','epsilon_cutoff_actives')
threshold_i = config.getfloat('TRAINING_PARAMETERS','epsilon_cutoff_inactives')
discard_inactives= False
similarity_thres= None


#STAT PARAMETERS
beta = config.getfloat('STAT_PARAMETERS','beta')
alpha = config.getfloat('STAT_PARAMETERS','alpha')


#FILTER
filter_pains = True
filter_properties = config.getboolean('FILTER','filter_properties')
molwt_min = config.getint('FILTER','molwt_min')
logp_min = config.getint('FILTER','logp_min')
hdonors_min = config.getint('FILTER','hdonors_min')
haccept_min = config.getint('FILTER','haccept_min')
rotabonds_min = config.getint('FILTER','rotabonds_min')
heavat_min = config.getint('FILTER','heavat_min')
molwt_max = config.getint('FILTER','molwt_max')
logp_max = config.getint('FILTER','logp_max')
hdonors_max = config.getint('FILTER','hdonors_max')
haccept_max = config.getint('FILTER','haccept_max')
rotabonds_max = config.getint('FILTER','rotabonds_max')
heavat_max = config.getint('FILTER','heavat_max')



# In[6]:


#CONFIG CHECKS
if use_chembl:
    if type(chembl_file) == list:
        for i in chembl_file:
            if not os.path.isfile(i):
                print(f'ERROR: The indicated ChEMBL csv file {i} does not exist')
                sys.exit()
    elif type(chembl_file) == str:
        if not os.path.isfile(chembl_file):
            print(f'ERROR: The indicated ChEMBL csv file {chembl_file} does not exist')
            sys.exit()

if use_external_actives:
    if type(actives_file) == list:
        for i in actives_file:
            if not os.path.isfile(i):
                print(f'ERROR: The indicated active database file {i} does not exist')
                sys.exit()
    elif type(actives_file) == str:
        if not os.path.isfile(actives_file):
            print(f'ERROR: The indicated active database file {actives_file} does not exist')
            sys.exit()

if use_external_inactives:
    if type(inactives_file) == list:
        for i in inactives_file:
            if not os.path.isfile(i):
                print(f'ERROR: The indicated inactive database file {i} does not exist')
                sys.exit()
    elif type(inactives_file) == str:
        if not os.path.isfile(inactives_file):
            print(f'ERROR: The indicated inactive database file {inactives_file} does not exist')
            sys.exit()

if mode=='screening' and False:
    if type(db_to_screen) == list:
        for i in db_to_screen:
            if not os.path.isfile(i):
                print(f'ERROR: The indicated database to screen file {i} does not exist')
                sys.exit()
    elif type(db_to_screen) == str:
        if not os.path.isfile(db_to_screen):
            print(f'ERROR: The indicated database file screen file {db_to_screen} does not exist')
            sys.exit()
else:            
            
    if use_external_decoys:
        if type(decoys_file) == list:
            for i in decoys_file:
                if not os.path.isfile(i):
                    print(f'ERROR: The indicated decoys database file {i} does not exist')
                    sys.exit()
        elif type(decoys_file) == str:
            if not os.path.isfile(decoys_file):
                print(f'ERROR: The indicated decoys database file {inactives_file} does not exist')
                sys.exit()

if fp_type == 'mhfp':
    mhfp_encoder = Chem.rdMHFPFingerprint.MHFPEncoder(nbits)
elif fp_type == 'avalon':
    from rdkit.Avalon import pyAvalonTools

    



# # DB preparation functions

# In[7]:


def bitjoiner(fp):
    strfp= []
    for i in (list(fp)):

        strfp.append(str(i))
    return "".join(strfp)

###############################################

def get_fingerprints_ecfp(df,string=True,keep_old=False,verbose=verbose,drop_zeros=True,fp_sim=False):
    
    #print('Preparing database...')
    start_time=time.time()
    counter= 0
    fps=[]
    fp_string=[]
    ligprepped_smiles=[]
    fps_sim=[]
    percentages=[20,40,60,80,100]
    
    smiles = df['Smiles']
    zero_fp=np.zeros(nbits)
    zero_fp = np.array([int(x) for x in list(zero_fp)])
    zero_fp_str = bitjoiner(zero_fp)
    l=len(df)
    print('Database Preparation: Stripping salts...')
    print('Database Preparation: Calculating tautomeric and protonation states for physiological pH...')
    if fp_type =='ecfp':
        print(f'Database Preparation: Converting in {fp_type.upper()} fingerprints, vectors of {nbits} bits with a radius of {iterations}...')
    elif (fp_type =='rdkit') or (fp_type == 'avalon'):
        print(f'Database Preparation: Converting in {fp_type.upper()} fingerprints, vectors of {nbits} bits...')
    elif fp_type =='tt':
        print(f'Database Preparation: Converting in Topological Torsions fingerprints, vectors of {nbits} bits...')
    else:
        print(f'Database Preparation: Converting in {fp_type.upper()} fingerprints, with {nbits} permutations and a radius of {iterations}...')
    
    
    def calculator(smile):
        nonlocal counter
        nonlocal percentages
        
        if verbose == False:
            rdBase.DisableLog('rdApp.error')
        
        df_title_column=df['Title']
        title_name=df_title_column.iloc[counter]    
        #print(title_name)
        try:
            try:
                mol = pybel.readstring("smi",smile)
                mol.OBMol.StripSalts()
                mol.OBMol.AddHydrogens(False,True)
                mol.OBMol.ConvertDativeBonds()
                new_smile= mol.write().rstrip()
                mol_rd = Chem.MolFromSmiles(new_smile)
                if explicit_hydrogens:
                    mol_rd=Chem.RemoveHs(mol_rd)
                    mol_rd=Chem.AddHs(mol_rd)
                if fp_type == 'ecfp':
                    fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol_rd, iterations, nBits = nbits, useChirality=chirality, useFeatures=features, includeRedundantEnvironments=redundancy )
                    fp1 = fp.ToBitString()
                    fp1 = np.array([int(x) for x in list(fp1)])                
                elif fp_type == 'rdkit':
                    fp = Chem.RDKFingerprint(mol_rd, fpSize=nbits)
                    fp1 = fp.ToBitString()
                    fp1 = np.array([int(x) for x in list(fp1)])
                elif fp_type == 'mhfp':
                    fp = mhfp_encoder.EncodeMol(mol_rd, radius= iterations, isomeric=chirality)
                    fp1 = np.array(fp)
                elif fp_type == 'tt':
                    fp=Torsions.GetHashedTopologicalTorsionFingerprint(mol_rd,nbits,includeChirality=chirality)
                    fp1 = np.array([int(x) for x in list(fp)])
                elif fp_type == 'avalon':
                    fp = pyAvalonTools.GetAvalonFP(mol_rd,nBits=nbits)
                    fp1 = np.array([int(x) for x in list(fp)])



                    
            except:
                if verbose:
                    print(f'Database Preparation: Trying to fix {title_name}')
                    print(f'-- {title_name} SMILES string: {smile}')
                mol = pybel.readstring("smi",smile)
                mol.OBMol.StripSalts()
                if "[C-]#[N+]" in smile:
                    mol.OBMol.AddHydrogens(False,True)
                else:                  
                    mol.OBMol.AddHydrogens(False,False)
                    mol.OBMol.ConvertDativeBonds()
                
                new_smile= mol.write().rstrip()
                mol_rd = Chem.MolFromSmiles(new_smile)
                if explicit_hydrogens:
                    mol_rd=Chem.RemoveHs(mol_rd)
                    mol_rd=Chem.AddHs(mol_rd)
                if fp_type == 'ecfp':
                    fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol_rd, iterations, nBits = nbits, useChirality=chirality, useFeatures=features, includeRedundantEnvironments=redundancy )
                    fp1 = fp.ToBitString()
                    fp1 = np.array([int(x) for x in list(fp1)])                
                elif fp_type == 'rdkit':
                    fp = Chem.RDKFingerprint(mol_rd, fpSize=nbits)
                    fp1 = fp.ToBitString()
                    fp1 = np.array([int(x) for x in list(fp1)])
                elif fp_type == 'mhfp':
                    fp = mhfp_encoder.EncodeMol(mol_rd, radius= iterations, isomeric=chirality)
                    fp1 = np.array(fp)
                elif fp_type == 'tt':
                    fp=Torsions.GetHashedTopologicalTorsionFingerprint(mol_rd,nbits,includeChirality=chirality)
                    fp1 = np.array([int(x) for x in list(fp)])
                elif fp_type == 'avalon':
                    fp = pyAvalonTools.GetAvalonFP(mol_rd,nBits=nbits)
                    fp1 = np.array([int(x) for x in list(fp)])
                    
                    


                if verbose:
                    print(f'-- {title_name} fixed with the following SMILES string {new_smile}')


            
        except:
            if verbose:
                print(f'WARNING: {title_name} COULD NOT be fixed and converted in fingerprints.')
            else:
                print(f'WARNING: {title_name} COULD NOT be converted in fingerprints.')
            new_smile= smile
            fp=np.zeros(nbits)
            fp1 = zero_fp
            
        fp_string.append(bitjoiner(fp1))
        fps.append(fp1)
        ligprepped_smiles.append(new_smile)
        fps_sim.append(fp)
        


        percentage = int(100*(counter/l))
        if percentage in percentages:
            if verbose:
                print(f'{(percentage):.0f}% done')
            percentages.remove(percentage)
            

        counter = counter +1
        return None
        
        
    smiles.apply(calculator)
    
    if fp_sim:
        df['fp_sim'] = fps_sim
    df['fp_string'] = fp_string
    df['fp'] = fps
    if keep_old==True:
        df['Old Smiles']=df['Smiles']

    df['Smiles'] = ligprepped_smiles
    if drop_zeros:
        df=df[df['fp_string']!=zero_fp_str]
    
    df.reset_index(inplace = True)
    df = df.drop(['index'], axis = 1)
    if not string:
        df = df.drop(['fp_string'], axis = 1)
    if verbose:
        print(f'Database loaded in {(time.time() - start_time):.2f} seconds')
    #print('Preparation complete')
    
    
    del fps
    del fp_string
    del ligprepped_smiles
    del fps_sim

    
    
    return df






######################################

def file_reader(filename):
    
    cols = 2
    
    def index_namer(name):
    
        new_name = (f'cmp_{str(name)}')
        return new_name

    
    def is_smile(items):

        if type(items) != list:
            if type(items) == str:
                temp=[]
                temp.append(items)
                items=temp
            else:
                items =list(items)
        for i in items:
            try:
                mol = pybel.readstring("smi",i)
                return True
            except:
                pass
        return False

    def column_title_smiles(row):
        for i in row:
            if 'smile' in i.lower():
                return i
        return False

    
    def recognizer(item):
        nonlocal cols
        if type(item)==pd.core.frame.DataFrame:
            df=item
            
        elif type(item)==str:
            try:
                df = pd.read_csv(item, sep=None)
                cols = len(df.columns)
            except Exception as mye:
                if 'fields' in str(mye):
                    df= pd.read_csv(item,header=None)
                    cols=1
                else:
                    print('ERROR: Could not recognize file format')
                    sys.exit()
            
        if cols==1:
            df.rename(columns={0:'Smiles'}, inplace=True)

            if not (is_smile(df.iloc[0][0])):
                df.drop(0,inplace=True)
                df.reset_index(inplace=True, drop=True)
            df['Title']=df.index
            df['Title']= df['Title'].apply(index_namer)
                
        else:        
                
            if 'Molecule ChEMBL ID' in (list(df.columns)):
                df.rename(columns={'Molecule ChEMBL ID':'Title'}, inplace=True)
        
            
            elif column_title_smiles(df.columns):
                if column_title_smiles(df.columns) != 'Smiles':
                    df.rename(columns={column_title_smiles(df.columns):'Smiles'}, inplace=True)
                for i in df.columns:
                    if 'title' in i.lower():
                        if i != 'Title':
                            df.rename(columns={i:'Title'}, inplace=True)
                        break
                if 'Title' not in (list(df.columns)):
                    for i in df.columns:
                        if i != 'Smiles':
                            df.rename(columns={i:'Title'}, inplace=True)
                            break
                    
                            
                    
            
            
            else:
                df = pd.read_csv(item,header=None,sep=None)
                exit=0
                for i in range(len(df)):
                    for j in range(len(df.columns)):
                        if is_smile(df.iloc[i][j]):
                            if j !=0:
                                df.rename(columns={j:'Smiles', 0:'Title'}, inplace=True)
                                exit=1
                            else:
                                df.rename(columns={0:'Smiles', 1:'Title'}, inplace=True)
                                exit=1
                            break
                    if exit==1:
                        break

        return df
    
    try:
        
        if type(filename) == list:
            print('Database preparation: Merging multiple entries in a single dataset...')
            for i in range(len(filename)):
                if i == 0:
                    df = recognizer(filename[i])
                else:
                    df_2 = recognizer(filename[i])
                    df = pd.concat([df,df_2],ignore_index=True)
            filename=df
        else:
            df=recognizer(filename)
                        
        return df    
    except UnboundLocalError:
        raise ValueError('Could not recognize file format')

#################################################
def load_decoys(filename,force_sample=True,sample_number=sample_number):
    df=file_reader(filename)
    
    original_decoy_num = len(df.iloc[:,0])
    
    df = df.sort_values(by=['Title'])
    df = df.drop_duplicates(subset = 'Title', keep = 'first')
    df = df.dropna(subset = ['Smiles'])
    if force_sample==True:
        if len(df['Title']) > sample_number:
            df = df.sample(n=sample_number) 
    df.reset_index(inplace = True)
    df = df.drop('index', axis = 1)
    df = get_fingerprints_ecfp(df)
    df = df.drop_duplicates(subset = 'fp_string', keep = 'first')
    df.reset_index(inplace = True)
    df = df.drop(['index','fp_string'], axis = 1)

    

        
    return original_decoy_num, df
#######################################################


#RETURNS INTER-SIMILARITY BETWEEN DATAFRAMES
def calculate_similarity(df_1,df_2,del_ones=False):
    
    
    df=df_1.copy()
    df2=df_2.copy()
    fp_2 = df2['fp_sim']
    fp1 = df['fp_sim']
    #print(fp1)
    def calc(fp):
        #print(fp)
        #to_delete.append(rdkit.DataStructs.FingerprintSimilarity(fp,fp_2))
        if fp_type == 'mhfp':
            similarity = fp_2.apply(mhfp_encoder.Distance, args=(fp,))
        
        elif (fp_type == 'tt') or (fp_type == 'avalon'):
            similarity = fp_2.apply(DataStructs.TanimotoSimilarity, args=(fp,))
        else:
            similarity = fp_2.apply(rdkit.DataStructs.FingerprintSimilarity, args=(fp,))
        
    
        #fp_2.drop(0,inplace=True)
        #fp_2.reset_index(inplace = True,drop= True)
        
        return similarity    
    matrix = fp1.apply(calc)
    df.rename(columns={'Title':'Inactives'}, inplace=True)
    df2.rename(columns={'Title':'Actives'}, inplace=True)
    matrix.index= df['Inactives']
    matrix.columns= df2['Actives']
    
    similarity= matrix.max(axis=1)
    similarity.reset_index(inplace=True,drop=True)
    
    most_similar= matrix.idxmax(axis=1)
    most_similar.reset_index(inplace=True,drop=True)

    
    df['similarity']=similarity
    df['most similar compound']=most_similar
    
    if del_ones==True:
        df=df[df['similarity']!=1]
        df.reset_index(inplace = True)
        df.drop(['similarity','most similar compound','index'], axis = 1, inplace=True)
    df.rename(columns={'Inactives':'Title'}, inplace=True)
    
    return df

############################################
######## PROPERTIES FILTER ###################

def prop_filter(df_1):
    
    df=df_1.copy()
    smiles = df['Smiles']
    def calc(smile):
        
        m = Chem.MolFromSmiles(smile)
        
        
        molwt = int(Descriptors.MolWt(m)) in range(molwt_min, molwt_max +1)
        logp = int(Descriptors.MolLogP(m)) in range(logp_min, logp_max +1)
        hdonors = Descriptors.NumHDonors(m) in range(hdonors_min,hdonors_max + 1)
        haccept = Descriptors.NumHAcceptors(m) in range(haccept_min,haccept_max +1)
        heavat = Descriptors.HeavyAtomCount(m) in range(heavat_min,heavat_max+1)
        rotabonds = Descriptors.NumRotatableBonds(m) in range(rotabonds_min,rotabonds_max+1)

        
        
        
        
        if  molwt and logp and hdonors and haccept and haccept and heavat and rotabonds == True:
            return 0
        else:
            return 1
    
    df['to_filter'] = smiles.apply(calc)
    df=df[df['to_filter'] == 0]
    df.reset_index(inplace=True,drop=True)
    df.drop(['to_filter'], axis = 1, inplace=True)
    
    return df

######## PAINS FILTER ###################

def pains_filter(df_1):
    
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
    catalog = FilterCatalog.FilterCatalog(params)

    df=df_1.copy()
    smiles = df['Smiles']
    def calc(smile):
        
        m = Chem.MolFromSmiles(smile)
        is_a_pain = (catalog.HasMatch(m))

        if is_a_pain:
            return 'Yes'
        else:
            return 'No'
    
    df['potential_pain'] = smiles.apply(calc)
    #df=df[df['potential_pain'] == 0]
    #df.reset_index(inplace=True,drop=True)
    #df.drop(['potential_pain'], axis = 1, inplace=True)
    
    return df


# In[8]:


#This function allows to load a CHEMBL csv, clean it and split it in active set (class 0), inactive set (class 1), and discarded compounds (class 2) for that we have either no data or an activity that is in the so-called grey area
def load_chembl_dataset(file_name, comment=False, gray= False):
    
    comment_uncertain_keywords = ['not determined','no data','nd(insoluble)','not evaluated','dose-dependent effect','uncertain','tde','inconclusive','active-partial']
    comment_inactive_keywords = ['not active','inactive']

    df=file_reader(file_name)
    
    print(f'''\nTraining: Compounds will be considered active if they are reported to have a value of 
          IC50, EC50, Ki, Kd, or potency, inferior to {activity_threshold} nM. 
          They will be classified as inactive if their IC50, EC50, Ki, Kd, or potency will be greater
          than {inactivity_threshold} nM, or if their inhibition rate is lower than {inhibition_threshold}%.
          For duplicate entries, only the most active one will be considered.\n''')
    
    inactive_types = ['inhibition']
    active_types = ['ec50','ic50','ki','kd','potency', 'kd apparent']
    class_list = []
    df = df.dropna(subset = ['Smiles','Standard Type'])

    df['Standard Value'] = pd.to_numeric(df['Standard Value'],errors = 'coerce')
    df['Standard Value'].fillna('NA', inplace = True)


    df.reset_index(inplace = True)
    df = df.drop('index', axis = 1)
    
    assays_list =[]
    if comment:
        comment_list=[]
    for i in range(len(df['Standard Type'])):

        
        c = df.loc[i, 'Comment']
        if type(c) == int:
            c = 'number'
            
        elif type(c) == str:
            #detect if a number is present in the string
            if not re.search('\d+', c):
                c = df.loc[i, 'Comment'].lower()
            else:
                #numbers present
            
                #c = df.loc[i, 'Comment'].lower().split()
                c = 'number'
        else :
            c = 'number'
        if comment:
            if c not in comment_list:
                comment_list.append(c)
        t = df.loc[i,'Standard Type'].lower()
        try:
            v = float(df.loc[i,'Standard Value'])
        except:
            v = (df.loc[i,'Standard Value'])
        u = df.loc[i,'Standard Units']
        r = df.loc[i,'Standard Relation']
        
        if t not in assays_list:
            assays_list.append(t)
        
        if c in comment_uncertain_keywords:
            class_list.append(2)
        
        #UNCOMMENT TO CONSIDER INACTIVE KEYWORDS 
        #elif c in comment_inactive_keywords:
        #   class_list.append(1)
        
        elif v == 'NA':
            class_list.append(2)
        
        
        elif (t in active_types) and (u == 'nM') and (v < activity_threshold) and (r != '\'>\'') :
            class_list.append(0)
        
        # CONSIDERS ALL THE "Inhibition" type with a "<" automatically inactive
        elif (t in inactive_types) and (((v<inhibition_threshold) and (u == '%' )) or (r == '\'<\'' )):
            class_list.append(1)
        # DOES NOT CONSIDER ALL THE "Inhibition" type with a "<" automatically inactive
        #elif (t in inactive_types) and (((v<inhibition_threshold) and (u == '%' )):
        #    class_list.append(1)

        
        elif (t in active_types) and (v > inactivity_threshold ):
            class_list.append(1)

####### UNCOMMENT TO CONSIDER ALL the ">" automatically inactive
        #elif (t in active_types) and (r == '\'>\'' ):
            #class_list.append(1)
        
        elif (t in active_types) and (v < inactivity_threshold ):
            class_list.append(2)


            
        else:
            class_list.append(2)
    df['class'] = class_list

    
    
    to_keep = ['Title','Smiles','Standard Value','Standard Type','Standard Relation', 'Standard Units','class','Comment']
    for i in dict(df).keys():
        if i not in to_keep:
            df = df.drop(i,axis=1)

    df = df.sort_values(by=['Title','class', 'Standard Value'])
    df = df.drop_duplicates(subset = 'Title', keep = 'first')
    
    df = df.dropna(subset = ['Smiles'])
    df.reset_index(inplace = True)
    df = df.drop('index', axis = 1)

    df=get_fingerprints_ecfp(df,fp_sim=True)
    if gray:
        df_gray = df[df['class'] == 2]
        df_gray.reset_index(inplace = True,drop = True)
    df = df[df['class'] != 2]
    df.reset_index(inplace = True)
    df = df.drop('index', axis = 1)

    
    #df = df.sort_values(by=['class'])
    #df = df.drop_duplicates(subset = 'fp_string', keep = 'first')

    #df.reset_index(inplace = True)
    #df = df.drop(['index','fp_string'], axis = 1)
    print('Database preparation: Removing duplicate entries...')
    if comment:
        return df, assays_list, comment_list
    elif gray:
        return df, df_gray
    else:
        return df


# # Algorithm Functions

# In[9]:


def scaler_light(X_train,y_train):
    
    training_a = X_train[y_train==0]
    training_i = X_train[y_train==1]

    scaler_a = StandardScaler().fit(training_a)
    scaler_i = StandardScaler().fit(training_i)
   
    training_a = scaler_a.transform(training_a)
    training_i = scaler_i.transform(training_i)

    active_columns_del = []
    for i in range(training_a.shape[1]):
        if training_a[:,i].std()==0.0:
            active_columns_del.append(i)

    inactive_columns_del = []
    for i in range(training_i.shape[1]):
        if training_i[:,i].std()==0.0:
            inactive_columns_del.append(i)


    training_a = np.delete(training_a, active_columns_del, axis = 1)
    training_i = np.delete(training_i, inactive_columns_del, axis = 1)
    
    return training_a, training_i

#########################################################


def scaler_external(X_train,y_train,ligands):
    
    training_a = X_train[y_train==0]
    training_i = X_train[y_train==1]


    scaler_a = StandardScaler().fit(training_a)
    scaler_i = StandardScaler().fit(training_i)

    training_a = scaler_a.transform(training_a)
    training_i = scaler_i.transform(training_i)

    ligands_activescaled = scaler_a.transform(ligands)
    ligands_inactivescaled = scaler_i.transform(ligands)



    active_columns_del = []
    for i in range(training_a.shape[1]):
        if training_a[:,i].std()==0.0:
            active_columns_del.append(i)

    inactive_columns_del = []
    for i in range(training_i.shape[1]):
        if training_i[:,i].std()==0.0:
            inactive_columns_del.append(i)



    training_a = np.delete(training_a, active_columns_del,axis = 1)
    training_i = np.delete(training_i, inactive_columns_del,axis = 1)
    
    ligands_activescaled = np.delete(ligands_activescaled, active_columns_del,axis = 1)
    ligands_inactivescaled = np.delete(ligands_inactivescaled, inactive_columns_del,axis = 1)
    return ligands_activescaled, ligands_inactivescaled
#########################################

def scaler(X_train,X_test,y_train,y_test):
    
    training_a, testset_a, training_a_y, testset_a_y = X_train[y_train==0],X_test[y_test==0],y_train[y_train==0],y_test[y_test==0]
    training_i, testset_i, training_i_y, testset_i_y = X_train[y_train==1],X_test[y_test==1],y_train[y_train==1],y_test[y_test==1]



    scaler_a = StandardScaler().fit(training_a)
    scaler_i = StandardScaler().fit(training_i)

    
    training_a = scaler_a.transform(training_a)
    training_i = scaler_i.transform(training_i)

    testset_a_activescaled = scaler_a.transform(testset_a)
    testset_a_inactivescaled = scaler_i.transform(testset_a)

    testset_i_activescaled = scaler_a.transform(testset_i)
    testset_i_inactivescaled = scaler_i.transform(testset_i)




    active_columns_del = []
    for i in range(training_a.shape[1]):
        if training_a[:,i].std()==0.0:
            active_columns_del.append(i)

    inactive_columns_del = []
    for i in range(training_i.shape[1]):
        if training_i[:,i].std()==0.0:
            inactive_columns_del.append(i)


    training_a = np.delete(training_a, active_columns_del,axis = 1)
    testset_a_activescaled = np.delete(testset_a_activescaled, active_columns_del,axis = 1)
    testset_a_inactivescaled = np.delete(testset_a_inactivescaled, inactive_columns_del,axis = 1)


    training_i = np.delete(training_i, inactive_columns_del,axis = 1)
    testset_i_activescaled = np.delete(testset_i_activescaled, active_columns_del,axis = 1)
    testset_i_inactivescaled = np.delete(testset_i_inactivescaled, inactive_columns_del,axis = 1)
    return training_a,testset_a_activescaled,testset_a_inactivescaled,training_i,testset_i_activescaled,testset_i_inactivescaled


# In[10]:


##########Core RMD Algorithm#############

def proj_vect(mu, num_eig, p, vv):
    mu_proj = np.zeros(p)
    for counter in range(num_eig):
        vect_prod = float(np.vdot(vv[:,counter], mu))
        mu_proj = mu_proj +vect_prod * vv[:,counter]
    return np.array([float(i) for i in mu_proj])

class RMDClassifier(object):
    
    def __init__(self, threshold = threshold, threshold_i =threshold_i):
        self.threshold = threshold
        self.epsilon = None
        self.vv = None
        self.num_eig = None
        self.p = None
        self.threshold_i = threshold_i
        self.epsilon_i = None
        self.vv_i = None
        self.num_eig_i = None
        self.p_i = None

    def fit(self, training_set):
        n,p = training_set.shape
        self.p = p
        covar = np.dot(training_set.T, training_set)/n
        eigen_values, eigen_vectors = np.linalg.eig(covar)
        counter = eigen_values.argsort()
        counter = counter[::-1]
        eigen_values = eigen_values[counter]
        eigen_vectors = eigen_vectors[:,counter]
        MP_bound = (1+np.sqrt(p/n))**2
        num_eig = eigen_values[eigen_values > MP_bound].shape[0]
        vv = eigen_vectors[:,:num_eig]
        self.num_eig, self.vv = num_eig, vv
        euc_distance = []
        for mol_vec_counter in range(n):
            mu = training_set[mol_vec_counter,:]
            mu_proj = proj_vect(mu, num_eig, p, vv)
            euc_distance.append(np.linalg.norm(mu-mu_proj))
        euc_distance.sort()
        euc_distance = np.array(euc_distance)
        cutoff_counter = int(self.threshold * len(euc_distance))
        epsilon = euc_distance[cutoff_counter]
        self.epsilon = epsilon
        
    def fit_i(self, training_set):
        
        n,p = training_set.shape
        self.p_i = p
        covar = np.dot(training_set.T, training_set)/n
        eigen_values, eigen_vectors = np.linalg.eig(covar)
        counter = eigen_values.argsort()
        counter = counter[::-1]
        eigen_values = eigen_values[counter]
        eigen_vectors = eigen_vectors[:,counter]
        MP_bound = (1+np.sqrt(p/n))**2
        num_eig = eigen_values[eigen_values > MP_bound].shape[0]
        vv = eigen_vectors[:,:num_eig]
        self.num_eig_i, self.vv_i = num_eig, vv
        euc_distance = []
        for mol_vec_counter in range(n):
            mu = training_set[mol_vec_counter,:]
            mu_proj = proj_vect(mu, num_eig, p, vv)
            euc_distance.append(np.linalg.norm(mu-mu_proj))        
        euc_distance.sort()
        euc_distance = np.array(euc_distance)
        cutoff_counter = int(self.threshold_i * len(euc_distance))
        epsilon = euc_distance[cutoff_counter]
        self.epsilon_i = epsilon        

        
    def predict(self, unknown_set,unknown_set_i, score = False):

        unknown_set_euc_distance = []
        unknown_set_euc_distance_i = []

        for mol_vec_counter in range(unknown_set.shape[0]):
            mu = unknown_set[mol_vec_counter,:]
            mu_inactivescaled = unknown_set_i[mol_vec_counter,:]
            mu_proj = proj_vect(mu, self.num_eig, self.p, self.vv)
            mu_proj_i = proj_vect(mu_inactivescaled, self.num_eig_i, self.p_i, self.vv_i)
            unknown_set_euc_distance.append( np.linalg.norm(mu - mu_proj))
            unknown_set_euc_distance_i.append( np.linalg.norm(mu_inactivescaled - mu_proj_i))

                    
        predictions = []
        if score == False:
            for i in range(len(unknown_set_euc_distance)):
                x = unknown_set_euc_distance[i] 
                y = unknown_set_euc_distance_i[i] 
                if x >= self.epsilon:
                    predictions.append(0)
                elif y > self.epsilon_i:
                    predictions.append(1)
                elif (x-(self.epsilon)) < (y-(self.epsilon_i)):
                    predictions.append(1)
                else:
                    predictions.append(0)
            return np.array(predictions)
        
        
        ######## EXPERIMENTAL
        elif (mode == 'screening') and (not temporal):
            RMD_score = []
            for i in range(len(unknown_set_euc_distance)):
                x = unknown_set_euc_distance[i] 
                y = unknown_set_euc_distance_i[i] 
                if x >= (self.epsilon):
                    predictions.append(0)
                    #ORIGINAL
                    RMD_score.append(0)
                    #NEW
                    #RMD_score.append((x-(self.epsilon)) - (y-(self.epsilon_i)))

                elif y > self.epsilon_i:
                    predictions.append(1)
                    #ORIGINAL
                    RMD_score.append(x-(self.epsilon))
                    #NEW
                    #RMD_score.append((x-(self.epsilon)) - (y-(self.epsilon_i)))

                elif (x-(self.epsilon)) < (y-(self.epsilon_i)):
                    predictions.append(1)
                    RMD_score.append((x-(self.epsilon)) - (y-(self.epsilon_i)))
                else:
                    predictions.append(0)
                    RMD_score.append(0)
                    #RMD_score.append((x-(self.epsilon)) - (y-(self.epsilon_i)))
               
            df = pd.DataFrame()
            df['predictions'] = predictions
            df['RMD_score'] = RMD_score
            
            return df

        
        else:
            RMD_score = []
            for i in range(len(unknown_set_euc_distance)):
                x = unknown_set_euc_distance[i] 
                y = unknown_set_euc_distance_i[i] 
                if x >= (self.epsilon):
                    predictions.append(0)
                    #ORIGINAL
                    RMD_score.append(x - (self.epsilon))
                    #NEW
                    #RMD_score.append((x-(self.epsilon)) - (y-(self.epsilon_i)))

                elif y > self.epsilon_i:
                    predictions.append(1)
                    #ORIGINAL
                    RMD_score.append(x-(self.epsilon))
                    #NEW
                    #RMD_score.append((x-(self.epsilon)) - (y-(self.epsilon_i)))

                elif (x-(self.epsilon)) < (y-(self.epsilon_i)):
                    predictions.append(1)
                    RMD_score.append((x-(self.epsilon)) - (y-(self.epsilon_i)))
                else:
                    predictions.append(0)
                    #RMD_score.append(0)
                    RMD_score.append((x-(self.epsilon)) - (y-(self.epsilon_i)))

                    
                    
            df = pd.DataFrame()
            df['predictions'] = predictions
            df['RMD_score'] = RMD_score
            
            return df


# # Algorithm benchmarking functions

# In[11]:


#Let's calculate the ROC metrics and Precision-Recall Curve!
def get_auc(actives_preds,inactives_preds,rep,fold,plot_roc=False,plot_prc=False):
    actives_preds['real binder'] =1
    inactives_preds['real binder'] =0
    auc_df = pd.concat([actives_preds,inactives_preds])
    auc_df = auc_df.sort_values(by=['RMD_score'])
    auc_df.reset_index(inplace = True)
    auc_df = auc_df.drop('index', axis = 1)

    y_true = auc_df['real binder']
    y_scores = (auc_df['RMD_score'] * (-1))
    #sklearn.metrics.roc_auc_score(y_true, y_scores)


##############   ROC  ###########################
    fpr,tpr , _ = sklearn.metrics.roc_curve(y_true, y_scores, drop_intermediate=True)
    #fpr,tpr = get_tpr_fpr(auc_df)
    roc_auc= sklearn.metrics.auc(fpr, tpr)
    

    if plot_roc==True:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='Area Under the Curve = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        #plt.title(f'Rep {rep} Fold {fold} ROC')
        plt.title(f'ROC curve')
        plt.legend(loc="lower right")
        plt.savefig(f'ROC_curve.png', dpi=300)
        #plt.savefig(f'Rep_{rep}_Fold_{fold}_ROC.png', dpi=300)
        #print(fpr)
###################  PRC    ############################

    #ap=average_precision_score(y_true, y_scores)
    #print(f'AP = {ap}')
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    prc_auc=sklearn.metrics.auc(recall, precision)
    #print(f'PRC AUC = {prc_auc}')
    
    if plot_prc==True:
        plt.figure()
        lw=2
        plt.plot(recall, precision, color='orangered',lw=lw, label='Area Under the Curve = %0.2f)' % prc_auc)
        #no_skill = len(y_true[y_true==1]) / len(y_true)
        #plt.plot([0, 1], [no_skill, no_skill], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim(top=1.0)  
        plt.title(f'PRC curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="lower left")
        plt.savefig(f'PRC_curve.png', dpi=300)
        
        
        
    return roc_auc, prc_auc

###################################################

def get_precision_stats(actives_preds,inactives_preds,beta=beta):
    
    actives_preds['real binder'] =1
    inactives_preds['real binder'] =0
    df = pd.concat([actives_preds,inactives_preds])
    df = df.sort_values(by=['RMD_score'])
    df.reset_index(inplace = True,drop=True)

    y_true = df['real binder']
    y_pred = df['predictions']
    
    
    
    number_of_tp=len(actives_preds[actives_preds['predictions']==1])
    number_of_fp=len(inactives_preds[inactives_preds['predictions']==1])
    try:
        precision= number_of_tp/(number_of_tp+number_of_fp)
    except:
        precision = 0
    f_score=sklearn.metrics.fbeta_score(y_true, y_pred, beta =beta)
    
    #### BEDROC

    
    scores= np.array(df.drop('predictions', axis = 1))
    bedroc= Scoring.CalcBEDROC(scores, 1,alpha)
    
    
    return precision, f_score, bedroc
    


######################################################
def get_confidence_interval(a):
    
    lower,upper=(st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a)))
    #print(sms.DescrStatsW(a).tconfint_mean())
    #print(st.norm.interval(0.95, loc=np.mean(a), scale=st.sem(a)))
    estimate= (lower+upper)/2
    margin=upper-estimate
    print(f'~~ {estimate:.3f} ± {margin:.3f}')
    return estimate, margin
#################################################

def get_fold_number(index):
    
    #index=((rep-1)*10)+fold-1
    if len(str(index)) == 2:
        rep=int(str(index)[0])+1
        fold=int(str(index)[1])+1
    else:
        rep=1
        fold=int(str(index)[0])+1
    #print(f'Rep {rep} Fold {fold}')
    return rep,fold


# # EXECUTION

# In[12]:


def list_2_string(prog_input):
    if type(prog_input) == list:
        return (' ').join(prog_input)
    else:
        return prog_input
        


# In[13]:


#Databases loading
print('''
**************************************************************************************
* PyRMD: AI-powered Virtual Screening                                                *
* Copyright (C) 2021  Dr. Giorgio Amendola, Prof. Sandro Cosconati                   *
*                                                                                    *
* This program is free software: you can redistribute it and/or modify               *
* it under the terms of the GNU Affero General Public License as published           * 
* by the Free Software Foundation, either version 3 of the License, or               *
* (at your option) any later version.                                                *
*                                                                                    *
* This program is distributed in the hope that it will be useful,                    *
* but WITHOUT ANY WARRANTY; without even the implied warranty of                     *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                      * 
* GNU Affero General Public License for more details.                                * 
*                                                                                    *
* You should have received a copy of the GNU Affero General Public License           *
* along with this program.  If not, see <https://www.gnu.org/licenses/>.             *
*                                                                                    * 
* If you used PyRMD in your work, or you require information regarding the software, *
* please email the authors at                                                        *
* giorgio.amendola@unicampania.it --- sandro.cosconati@unicampania.it                *
*                                                                                    *
* Please check our GitHub page for more information                                  *
* <https://github.com/cosconatilab/PyRMD>                                            *
**************************************************************************************

''')
print('''                                                                                
                                                                                
                                                                                
                                                                                
                                       (((                                      
                                      (((((                                     
                                    ((((((((                                    
                                   (((((((((                                    
                                 ((((((((((   %%                                
                                ((((((((((   %%%%                               
                              ####((((((   %%%%%%%%                             
                            (#########(    %%%%%%%%%                            
                           ##########       %%%%%%%%%%                          
                         ###########         %%%%%%#####                        
                        %%%#######             ##########                       
                      %%%%%%%%%##               ###########                     
                     (%%%%%%%%%                   ##########                    
                   (((%%%%%%%(                     #########((                  
                  ((((#%%%%%                         ##((((((((                 
                 (((((##%%%                           %(((((((((                
                 (((#########                       %%%%((((((((                
                  (##########%%%%               %%%%%%%%(((((((                 
                    #######%%%%%%%%%%&     #####%%%%%%%((((((                   
                        #%%%%%%%%     (###########%%%%(((                       
                                 ,((((((###########%                            
                                 (((((((((#####/                                
                                       (((                                      
                                                                                
                                                                                
                                                                                
        &&&&&&&&&&            &&&&&&&&&&   (&&&       &&&&   &&&%&&&&&&         
        &&&&  &&&& &&&   &&&  &&&&  /&&&   &&&&&.    &&&&&   &&&#   &&&&        
        %&&&%&&&%  &%&   &&&  &&%&&&%&&    &%&&&%   %&&&%&   &&&(    &&&%       
        &&&&       &&&   &&&  &&&&  &&&&   &&& &&& &&& &&&   &&&#    &&&        
        &&&&       &&&&&&&&&, &&&&   &&&&  &&&  &&&&&  &&&&  &&&%&&&&&&         
                          &&&                                                   
                    &&&&&&&&                                                    
 
 
            +--------------------------------------------------+
            |       PyRMD: AI-powered Virtual Screening        |
            |                                                  |
            |            ~~~~ Cosconati Lab ~~~~               |
            +--------------------------------------------------+
\n''')

if mode == 'benchmark':
    print('//////// BENCHMARK MODE \\\\\\\\\\\\\\\\ ')
elif mode == ('screening'):
    print('//////// SCREENING MODE \\\\\\\\\\\\\\\\ ')

else:
    print('Please indicate a valid mode in the configuration file')
    sys.exit()
    
    
print('\n----BUILDING TRAINING DATASET----\n')

if use_chembl:
    print(f'Training: Loading the ChEMBL dataset(s): {list_2_string(chembl_file)}')    
    if gray:
        df_training, df_gray =load_chembl_dataset(chembl_file, gray =True)
        df_gray = df_gray.drop(['fp','fp_sim','fp_string'], axis = 1)
        df_gray.to_csv(path_or_buf= 'chembl_discarded.csv',index=False)
        print('Training: Discarded compounds from the ChEMBL database have been written to chembl_discarded.csv')
    else:
        df_training =load_chembl_dataset(chembl_file)
    if use_external_actives:
        print(f'Training: Loading the active compounds dataset(s): {list_2_string(actives_file)}')
        df_actives=file_reader(actives_file)
        df_actives=get_fingerprints_ecfp(df_actives)
        df_actives['class']= 0
        df_training = pd.concat([df_training,df_actives],ignore_index=True)
    if use_external_inactives:
        print(f'Training: Loading the inactive compounds dataset(s): {list_2_string(inactives_file)}')
        df_inactives=file_reader(inactives_file)
        df_inactives=get_fingerprints_ecfp(df_inactives)
        df_inactives['class']= 1
        df_training = pd.concat([df_training,df_inactives],ignore_index=True)
    df_training = df_training.sort_values(by=['class'])
    df_training = df_training.drop_duplicates(subset = 'fp_string', keep = 'first')
    df_training.reset_index(inplace = True)
    df_training = df_training.drop(['index','fp_string'], axis = 1)

elif use_external_actives:
    print(f'Training: Loading the active compounds dataset(s): {list_2_string(actives_file)}')
    if use_external_inactives:
        print(f'Training: Loading the inactive compounds dataset(s): {list_2_string(inactives_file)}')
        df_actives=file_reader(actives_file)
        df_inactives=file_reader(inactives_file)
        df_inactives['class']= 1
        df_actives['class']= 0
        df_training = pd.concat([df_inactives,df_actives],ignore_index=True)
        df_training= get_fingerprints_ecfp(df_training)
        df_training = df_training.sort_values(by=['class'])
        df_training = df_training.drop_duplicates(subset = 'fp_string', keep = 'first')
        df_training.reset_index(inplace = True)
        df_training = df_training.drop(['index','fp_string'], axis = 1)

    else:
        raise ValueError('If you are not using a CHEMBL database, both an active compounds database and an inactive compounds database are required')
else:
    raise ValueError('If you are not using a CHEMBL database, both an active compounds database and an inactive compounds database are required')

        
# Writing a clean and human-readable copy of the training dataset
df_to_write = df_training.copy()
if 'Comment' in list(df_to_write.columns):
    df_to_write = df_to_write.drop(['fp','Comment','fp_sim'], axis = 1)
else:
    df_to_write = df_to_write.drop(['fp','fp_sim'], axis = 1)

df_to_write.loc[df_to_write['class'] == 0, 'class'] = 'Active'
df_to_write.loc[df_to_write['class'] == 1, 'class'] = 'Inactive'
inactives_num=len(df_to_write[df_to_write['class'] == 'Inactive'])
actives_num=len(df_to_write[df_to_write['class'] == 'Active'])
print(f'Training: The training dataset consists of {actives_num} active compounds and {inactives_num} inactive compounds')
print('Training: The training dataset has been written to the file clean_training_db.csv')
df_to_write.to_csv(path_or_buf= 'clean_training_db.csv',index=False) 
print('Training dataset preparation is complete!\n')

decoy_num = 0
original_decoy_num = 0

#LOADING DECOYS
if (use_external_decoys== True) and (mode=='benchmark'):
    print('\n----BUILDING DECOY DATASET----\n')
    
    print(f'Decoys: Loading the decoy compounds dataset(s): {list_2_string(decoys_file)}')
    
    print('Decoys: Decoy compounds will only be used for benchmarking purposes, they will not be used to train the algorithm')
    original_decoy_num, df_decoys = load_decoys(decoys_file,force_sample=True)
    decoy_num = len(df_decoys.iloc[:,0])
    print(f'Decoys: The decoy dataset consists of {original_decoy_num} compounds. Duplicate compounds will be discarded')
    print('Decoys dataset preparation is complete!\n')
    
    
######SIMILARITY CALCULATIONS
df_actives=df_training[df_training['class'] ==0]
df_actives.reset_index(inplace = True,drop = True)

if inactives_similarity:
    print('Calculating pairwise similarity between the inactives data set and the actives')
    df_inactives=df_training[df_training['class'] ==1]
    df_inactives.reset_index(inplace = True,drop = True)
    df_inactives=calculate_similarity(df_inactives,df_actives)
    df_inactives = df_inactives.drop(['fp','fp_sim'], axis = 1)
    df_inactives.to_csv(path_or_buf= inactives_similarity_file ,index=False)
    print(f'The inactives similarity to the actives has been written to {inactives_similarity_file}')
    
#GENERATE FPS AND AFFINITIES
fingerprints_training = np.stack(df_training['fp'])
affinity_training = df_training['class']
if (use_external_decoys== True) and (mode=='benchmark'):
    fingerprints_decoys = np.stack(df_decoys['fp'])
#Stratified Kfold split
X, y = fingerprints_training, affinity_training
#skf = StratifiedKFold(n_splits=n_splits, shuffle=True) #NON-REPEATED VERSION
skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
Ksplit=skf.split(X, y)
#UNCOMMENT TO KNOW THE SPLIT IN THE SETS --- USUALLY UNNECESSARY
#for train, test in Ksplit:
#    print('train -  {}   |   test -  {}'.format(
#        np.bincount(y[train]), np.bincount(y[test])))


# In[14]:


if mode == 'benchmark':
###############ALGORITHM BENCHMARKING########################
    fold = 0
    rep = 1
    results = pd.DataFrame()
    tp = []
    fp = []
    fp_ext = []
    fp_tot = []

    roc_aucs=[]
    prc_aucs=[]
    training_sets=[]
    test_sets=[]
    
    precisions=[]
    f_scores=[]
    bedrocs=[]
    
    print('\n----BENCHMARKING CALCULATIONS----\n')

    print(f'Benchmarking: Calculating statistics for {n_splits} training database splits and {n_repeats} repetition(s):')
    for train, test in Ksplit:
        gc.collect()
        training_sets.append(train)
        test_sets.append(test)
        X_train=X[train]
        y_train=y[train]
        X_test=X[test]
        y_test=y[test]
        training_a,testset_a_activescaled,testset_a_inactivescaled,training_i,testset_i_activescaled,testset_i_inactivescaled=scaler(X_train,X_test,y_train,y_test)
        clf = RMDClassifier()
        clf.fit(training_a)
        clf.fit_i(training_i)
        actives_preds = clf.predict(testset_a_activescaled, testset_a_inactivescaled, score=True )
        inactives_preds = clf.predict(testset_i_activescaled, testset_i_inactivescaled, score =True)

        fold = fold +1
        tp.append(np.mean(actives_preds['predictions']))
        fp.append(np.mean(inactives_preds['predictions']))
        if use_external_decoys==True:
            ext_decoys_activescaled,ext_decoys_inactivescaled=scaler_external(X_train,y_train,fingerprints_decoys)
            ext_decoys_preds=clf.predict(ext_decoys_activescaled,ext_decoys_inactivescaled,score=True)
            fp_ext.append(np.mean(ext_decoys_preds['predictions']))
            inactives_preds = pd.concat([inactives_preds,ext_decoys_preds],ignore_index=True)
            fp_tot.append(np.mean(inactives_preds['predictions']))

        roc_auc, prc_auc=get_auc(actives_preds,inactives_preds,rep,fold)
        roc_aucs.append(roc_auc)
        prc_aucs.append(prc_auc)
        precision, f_score, bedroc= get_precision_stats(actives_preds,inactives_preds)
        precisions.append(precision)
        f_scores.append(f_score)
        bedrocs.append(bedroc)
        print(f'\nRep {rep} Fold {fold}:')
        print(f'''
~~ True positive rate (Recall): {np.mean(actives_preds['predictions']):.3f}
~~ False positives rate: {np.mean(inactives_preds['predictions']):.3f}
~~ Precision: {precision:.3f}
~~ F Score: {f_score:.3f}
~~ ROC AUC: {roc_auc:.3f}
~~ PRC AUC: {prc_auc:.3f}
~~ BEDROC: {bedroc:.3f}
''')
        if fold ==n_splits:
            fold=0
            rep = rep+1
        
        ######## ATTEMPTS AT RAM CLEANING
        del training_a,testset_a_activescaled,testset_a_inactivescaled,training_i,testset_i_activescaled,testset_i_inactivescaled
        del actives_preds, inactives_preds
        if use_external_decoys==True:
            del ext_decoys_activescaled,ext_decoys_inactivescaled, ext_decoys_preds
            
    results['True Positives Rate'] = tp
    results['ROC AUC'] = roc_aucs
    results['PRC AUC'] = prc_aucs
    results['Precision'] = precisions
    results['F Score'] =f_scores
    results['BEDROC'] = bedrocs
    if use_external_decoys == True:
        results['Internal False Positives Rate'] = fp
        results['External False Positives Rate'] = fp_ext
        results['False Positives Rate'] = fp_tot
    else:
        results['False Positives Rate'] = fp


# In[15]:


####CONFIDENCE INTERVAL
if mode == 'benchmark':
    
    print('''\n+---------------------------------------------------------+
| Averaged Benchmarking Results With Confidence Intervals |
+---------------------------------------------------------+
''')
    print(f'\nTrue Positive Rate (Recall)')
    tp_ci, tp_margin = get_confidence_interval(results['True Positives Rate'])
    print(f'\nFalse Positive Rate')
    fp_ci, fp_margin = get_confidence_interval(results['False Positives Rate'])
    print(f'\nPrecision')
    precision_ci, precision_margin = get_confidence_interval(results['Precision'])
    print(f'\nF Score')
    fscore_ci, fscore_margin = get_confidence_interval(results['F Score'])
    print(f'\nROC AUC')
    roc_ci, roc_margin = get_confidence_interval(results['ROC AUC'])
    print(f'\nPRC AUC')
    prc_ci, prc_margin = get_confidence_interval(results['PRC AUC'])
    print(f'\nBEDROC')
    bedroc_ci, bedroc_margin = get_confidence_interval(results['BEDROC'])


    print('\n+---------------------------------------------------------+')


# In[16]:


###### GET REPRESENTATIVE AUC CURVE
if mode == 'benchmark':
    
    print('\nBenchmarking: Generating representative ROC curve')
    best_diff =100000
    for i in range(len(results['ROC AUC'])):
        auc = results['ROC AUC'][i]
        if (abs(auc-roc_ci)) < best_diff:
            best_diff=abs(auc-roc_ci)
            best_i = i
            best_rep, best_fold= get_fold_number(i)
            best_fold_auc= results['ROC AUC'][i]
    #print(f'The estimated AUC is: {auc_ci:.2f}, and the fold closest to the estimated value is Rep {best_rep} Fold {best_fold:} (index {best_i}) with an AUC of {best_fold_auc:.2f} and a difference of {best_diff:.2f})')
    train=training_sets[best_i]
    test=test_sets[best_i]

    X_train=X[train]
    y_train=y[train]
    X_test=X[test]
    y_test=y[test]
    training_a,testset_a_activescaled,testset_a_inactivescaled,training_i,testset_i_activescaled,testset_i_inactivescaled=scaler(X_train,X_test,y_train,y_test)
    clf = RMDClassifier()
    clf.fit(training_a)
    clf.fit_i(training_i)
    actives_preds = clf.predict(testset_a_activescaled, testset_a_inactivescaled, score=True )
    inactives_preds = clf.predict(testset_i_activescaled, testset_i_inactivescaled, score =True)

    if use_external_decoys==True:
        ext_decoys_activescaled,ext_decoys_inactivescaled=scaler_external(X_train,y_train,fingerprints_decoys)
        ext_decoys_preds=clf.predict(ext_decoys_activescaled,ext_decoys_inactivescaled,score=True)
        inactives_preds = pd.concat([inactives_preds,ext_decoys_preds],ignore_index=True)

    roc_auc, prc_auc=get_auc(actives_preds,inactives_preds,best_rep,best_fold,plot_roc=True)
    
    #print(f'Rep {best_rep} Fold {best_fold}:')
    #print(f"True positives rate: {np.mean(actives_preds['predictions'])}\nFalse positives rate: {np.mean(inactives_preds['predictions'])},\nAUC: {auc_ci}")
    
    print('Benchmarking: ROC curve saved to ROC_curve.png')


# In[17]:


###### GET REPRESENTATIVE PRC CURVE 
if mode == 'benchmark':
    
    print('\nBenchmarking: Generating representative PRC curve')
    best_diff =100000
    for i in range(len(results['PRC AUC'])):
        auc = results['PRC AUC'][i]
        if (abs(auc-prc_ci)) < best_diff:
            best_diff=abs(auc-prc_ci)
            best_i = i
            best_rep, best_fold= get_fold_number(i)
            best_fold_auc= results['PRC AUC'][i]
    #print(f'The estimated AUC is: {auc_ci:.2f}, and the fold closest to the estimated value is Rep {best_rep} Fold {best_fold:} (index {best_i}) with an AUC of {best_fold_auc:.2f} and a difference of {best_diff:.2f})')
    train=training_sets[best_i]
    test=test_sets[best_i]

    X_train=X[train]
    y_train=y[train]
    X_test=X[test]
    y_test=y[test]
    training_a,testset_a_activescaled,testset_a_inactivescaled,training_i,testset_i_activescaled,testset_i_inactivescaled=scaler(X_train,X_test,y_train,y_test)
    clf = RMDClassifier()
    clf.fit(training_a)
    clf.fit_i(training_i)
    actives_preds = clf.predict(testset_a_activescaled, testset_a_inactivescaled, score=True )
    inactives_preds = clf.predict(testset_i_activescaled, testset_i_inactivescaled, score =True)

    if use_external_decoys==True:
        ext_decoys_activescaled,ext_decoys_inactivescaled=scaler_external(X_train,y_train,fingerprints_decoys)
        ext_decoys_preds=clf.predict(ext_decoys_activescaled,ext_decoys_inactivescaled,score=True)
        inactives_preds = pd.concat([inactives_preds,ext_decoys_preds],ignore_index=True)

    roc_auc=get_auc(actives_preds,inactives_preds,best_rep,best_fold,plot_prc=True)
    
    #print(f'Rep {best_rep} Fold {best_fold}:')
    #print(f"True positives rate: {np.mean(actives_preds['predictions'])}\nFalse positives rate: {np.mean(inactives_preds['predictions'])},\nAUC: {auc_ci}")
    
    print('Benchmarking: PRC curve saved to PRC_curve.png')
    print('Benchmarking process complete!')


# In[18]:


############ SAVE BENCHMARK INFO IN A CSV
if mode == 'benchmark':

    benchmark = pd.DataFrame()

    benchmark['chembl_file']= [chembl_file]
    benchmark['activity_threshold']=[activity_threshold] 
    benchmark['inactivity_threshold']=[inactivity_threshold] 
    benchmark['inhibition_threshold']=[inhibition_threshold]
    benchmark['epsilon_cutoff_actives']=[threshold]
    benchmark['epsilon_cutoff_inactives']=[threshold_i]
    
    benchmark['Training Actives']= [actives_num]
    benchmark['Training Inactives']= [inactives_num]
    benchmark['Initial Number of Decoys'] = [original_decoy_num]
    benchmark['Number of Decoys'] = [decoy_num]
    

    
    benchmark['TPR'] = [tp_ci]
    benchmark['TPR CI'] = [tp_margin]

    benchmark['FPR'] = [fp_ci]
    benchmark['FPR CI'] = [fp_margin]

    benchmark['Precision'] = [precision_ci]
    benchmark['Precision CI'] = [precision_margin]

    benchmark['F Score'] = [fscore_ci]
    benchmark['F Score CI'] = [fscore_margin]

    benchmark['ROC AUC'] = [roc_ci]
    benchmark['ROC AUC CI'] = [roc_margin]

    benchmark['PRC AUC'] = [prc_ci]
    benchmark['PRC AUC CI'] = [prc_margin]

    benchmark['BEDROC'] = [bedroc_ci]
    benchmark['BEDROC CI'] = [bedroc_margin]


    #TRAINING DATASETS
    benchmark['actives_file']= [actives_file]
    benchmark['use_inactives']= [inactives_file]

    #Fingerprints Parameters
    benchmark['fp_type']= [fp_type]
    benchmark['explicit_hydrogens']=[explicit_hydrogens] 
    benchmark['iterations']=[iterations] 
    benchmark['nbits']=[nbits] 
    benchmark['chirality']=[chirality] 
    benchmark['redundancy']=[redundancy]
    benchmark['features']=[features]

    #DECOYS
    benchmark['use_decoys']=[use_external_decoys]
    benchmark['decoys_file']=[decoys_file]
    benchmark['sample_number']=[sample_number]

    #CHEMBL THRESHOLDS

    #KFOLD PARAMETERS
    benchmark['n_splits']=[n_splits] 
    benchmark['n_repeats']=[n_repeats]

    #STAT PARAMETERS
    benchmark['beta']=[beta]
    benchmark['alpha']=[alpha]

    
    print(f'The benchmark results are being written in the file {benchmark_file}')
    if Path(benchmark_file).is_file():
        benchmark.to_csv(path_or_buf= benchmark_file,index=False,mode='a', header=False)
    else:
        benchmark.to_csv(path_or_buf= benchmark_file,index=False)


# In[19]:


###############~~~ EXTERNAL DB PREDICTION FUNCTION ~~~#################
def ML_prediction(filename, final_name, X=X, y=y,score=score, decoy=False,sample_number=sample_number):
    
    print('\n----VIRTUAL SCREENING----\n')
    
    percentages= list(range(5,100,5))
    df=file_reader(filename)


    df = df.sort_values(by=['Title'])
    df = df.drop_duplicates(subset = 'Title', keep = 'first')
    df = df.dropna(subset = ['Smiles','Title'])
    if decoy==True:
        if len(df['Title']) > sample_number:
            df = df.sample(n=sample_number) 
    df.reset_index(inplace = True)
    df = df.drop('index', axis = 1)
    
    number_of_compounds = len(df['Smiles'])
    if number_of_compounds > 20000:
        number_of_chunks = int(number_of_compounds/10000)
        
    else:
        number_of_chunks =1
    l=number_of_chunks
    
    print(f'Screening: Preparing to screen a database with {number_of_compounds} entries')
    if number_of_chunks >1:
        print(f'Screening: Database will be split in {number_of_chunks} chunks')
    
    training_a,training_i=scaler_light(X,y)

    clf = RMDClassifier()
    clf.fit(training_a)
    clf.fit_i(training_i)

        
    count = 1
    
        
    def scorer(df_chunk):


        gc.collect()

        start_time = time.time()

        df_copy=df_chunk.copy()

        print(f'\nScreening: Processing chunk {count} out of {number_of_chunks}')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        df_copy=get_fingerprints_ecfp(df_copy,verbose=verbose,drop_zeros=False,string=False,fp_sim=True)
        print(f'Screening: Calculating predictions for chunk {count}....')
        smiles_chunk = df_copy['Smiles']
        chunk = np.stack(df_copy['fp'])
        chunk_b,chunk_d=scaler_external(X,y,chunk)

        if score == True: 
            df_preds = clf.predict(chunk_b,chunk_d, score=True)
            df_copy['predicted binder'] = df_preds['predictions']
            df_copy['RMD_score'] = df_preds['RMD_score'] *-1

        else:
            predictions=(clf.predict(chunk_b,chunk_d))
            df_copy['predicted binder'] = predictions

        ######### EXPERIMENTAL
        if (gray) or (inactives_similarity) or (temporal):
            pass
        else:
            df_copy=df_copy[df_copy['predicted binder'] == 1]
        df_copy.reset_index(inplace = True,drop=True)
        if len(df_copy['Title']) != 0:
            #df_copy=calculate_similarity(df_copy,df_training,del_ones=False) CALCULATE SIMILARITY TO THE WHOLE TRAINING SET
            df_copy=calculate_similarity(df_copy,df_actives,del_ones=False)
        
        else:
            df_copy['similarity'] =np.nan
            df_copy['most similar compound'] =np.nan
        df_copy = df_copy.drop(['fp','fp_sim'], axis = 1)
        if count == 1:
            df_copy.to_csv(path_or_buf= final_name,index=False)
        elif len(df_copy['Title']) != 0:
            df_copy.to_csv(path_or_buf= final_name,index=False,mode='a', header=False)

        del df_copy
        del chunk
        del chunk_b
        del chunk_d
        if not score:
            del predictions
        else:
            del df_preds
        del smiles_chunk



        print(f'Chunk processed in {(time.time() - start_time):.2f} seconds')



        return 'Ha!'

    for df_chunk in np.array_split(df, number_of_chunks):
        scorer(df_chunk)
        count = count +1
            
    
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f'\nPredictions complete! The results have been written to the file {screening_output}')
    return 'Done!'
  
def smi_converter(filename):
    df= file_reader(filename)
    if len(df['Title']) ==0:
        print('No actives found!')
        return None
    if 'RMD_score' in df.columns:
        df = df.drop(['RMD_score'],axis=1)
    df = df.drop(['similarity','most similar compound'], axis = 1)
    df.to_csv(path_or_buf= 'predicted_actives.smi',index=False, header=False, sep='\t')
#########SDF GEN
    if sdf_results:
        cmdCommand = f'obabel -ismi predicted_actives.smi -osdf -h --gen3D -Opredicted_actives.sdf'

        process = subprocess.Popen(cmdCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()


# In[ ]:


#####EXTERNAL DB PREDICTION
if mode == 'screening':
    
    screen= ML_prediction(db_to_screen,screening_output,score=score)
    df_test=file_reader(screening_output)
    df_test = df_test.drop(['predicted binder'], axis = 1)
    if len(df_test['Title']) ==0:
        print('No actives found!')
        sys.exit()
    if filter_properties:
        print('Filtering compounds that do not match the specified criteria...')
        df_test=prop_filter(df_test)
    if filter_pains:
        print('Filtering PAINS...')
        df_test=pains_filter(df_test)
    
    if sdf_results:
        print('Generating .smi and .sdf files of the predicted active compounds...')
    else:
        print('Generating a .smi file of the predicted active compounds...')
    df_test.to_csv(path_or_buf=screening_output,index=False)
    smi_converter(screening_output)
    print('Process complete!')

sys.exit()


