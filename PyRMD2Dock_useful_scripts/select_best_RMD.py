import pandas as pd

input_file = input("Enter the name of the screening output file: ")
num_compounds = int(input("Enter the number of compounds to select: "))
output_file_final = input("Enter the name of the output file (e.g., best10K.csv): ")
output_file_ligands = input("Enter the name of the ligands output file (e.g., 10kligs.csv): ")
full_database = input("Enter the path to initial database: ")

pred = pd.read_csv(input_file)
sort = pred.sort_values("RMD_score", ascending=False)
best_compounds = sort.head(num_compounds)

best_compounds.to_csv(output_file_final, index=False)
full = pd.read_csv(full_database)
whole = pd.merge(full, best_compounds, how="inner", on="Ligand")
ligands = whole[["Ligand", "smiles"]]
ligands.to_csv(output_file_ligands, index=False)
