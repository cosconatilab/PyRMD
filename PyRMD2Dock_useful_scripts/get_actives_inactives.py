import csv
import pandas as pd

# Prompt the user for the DG_activity_threshold and DG_inactivity_threshold values
DG_activity_threshold_str = input("Enter the DG_activity_threshold values (space-separated): ")
DG_activity_thresholds = [float(threshold.replace(',', '.')) for threshold in DG_activity_threshold_str.split()]

DG_inactivity_threshold_str = input("Enter the DG_inactivity_threshold values (space-separated): ")
DG_inactivity_thresholds = [float(threshold.replace(',', '.')) for threshold in DG_inactivity_threshold_str.split()]

# Prompt the user for the CSV file name
csv_file_name = input("Enter the CSV file name: ")

# Prompt the user for the file name containing the 'smiles' column
smiles_file_name = input("Enter the file name containing the 'smiles' column: ")

# Read the 'smiles' file using pandas
smiles_df = pd.read_csv(smiles_file_name, delimiter=',')

# Convert the 'Ligand' column to a common data type (e.g., string)
smiles_df['Ligand'] = smiles_df['Ligand'].astype(str)

# Process each DG_activity_threshold and DG_inactivity_threshold combination separately
for DG_activity_threshold in DG_activity_thresholds:
    for DG_inactivity_threshold in DG_inactivity_thresholds:
        # Generate the output file names
        active_threshold_str = str(DG_activity_threshold).replace('-', '').replace('+', '')
        inactive_threshold_str = str(DG_inactivity_threshold).replace('-', '').replace('+', '')

        active_output_file_name = f"actives{active_threshold_str}.csv"
        inactive_output_file_name = f"inactives{inactive_threshold_str}.csv"

        # Read the CSV file using pandas
        csv_df = pd.read_csv(csv_file_name)

        # Convert the 'Ligand' column to a common data type (e.g., string)
        csv_df['Ligand'] = csv_df['Ligand'].astype(str)

        # Select the rows based on the thresholds
        active_df = csv_df[csv_df['lowest_binding_energy'] <= DG_activity_threshold]
        inactive_df = csv_df[csv_df['lowest_binding_energy'] >= DG_inactivity_threshold]

        # Merge the 'smiles' column based on the 'Ligand' column
        active_df = active_df.merge(smiles_df, how='left', on='Ligand')
        inactive_df = inactive_df.merge(smiles_df, how='left', on='Ligand')

        # Write the selected active lines to the output file
        active_df.to_csv(active_output_file_name, index=False)

        # Write the selected inactive lines to the output file
        inactive_df.to_csv(inactive_output_file_name, index=False)

        print(f"Selected lines for DG_activity_threshold={DG_activity_threshold} saved in {active_output_file_name}")
        print(f"Selected lines for DG_inactivity_threshold={DG_inactivity_threshold} saved in {inactive_output_file_name}")

