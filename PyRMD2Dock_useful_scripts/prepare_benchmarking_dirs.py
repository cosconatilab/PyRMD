import os
import shutil

# Prompt the user for the values of DG_activity_threshold and DG_inactivity_threshold
DG_activity_threshold = input("Enter the DG activity thresholds (space-separated): ").split()
DG_inactivity_threshold = input("Enter the DG inactivity threshold (space-separated): ").split()

# Prompt the user for epsilon_cutoff_active and epsilon_cutoff_inactive
epsilon_cutoff_active = input("Enter the values for epsilon_cutoff_active (space-separated): ").split()
epsilon_cutoff_inactive = input("Enter the values for epsilon_cutoff_inactive (space-separated): ").split()

# Prompt the user for the path to the template_config.ini file
template_config_file = input("Enter the path to the template_config.ini file: ")

# Read the template_config.ini file
with open(template_config_file, "r") as template_file:
    template_data = template_file.read()

# Create the directories for every combination of DG_activity_threshold and DG_inactivity_threshold
for a in DG_activity_threshold:
    for i in DG_inactivity_threshold:
        directory_name = f'bench_{abs(float(a))}_{abs(float(i))}'
        os.makedirs(directory_name)
        os.chdir(directory_name)

        # Copy the actives{a}.csv and inactives{i}.csv files to the directory
        shutil.copy(f'../actives{abs(float(a))}.csv', 'actives.csv')
        shutil.copy(f'../inactives{abs(float(i))}.csv', 'inactives.csv')

        # Create the configuration files for every combination of epsilon_cutoff_active and epsilon_cutoff_inactive
        for u in epsilon_cutoff_active:
            for z in epsilon_cutoff_inactive:
                config_data = template_data.replace("XXXX", u).replace("YYYY", z)

                config_filename = f'config_{u}_{z}.ini'
                with open(config_filename, 'w') as config_file:
                    config_file.write(config_data)

        os.chdir('..')

# Copy other required files to each directory if needed
# shutil.copy(source_file, destination_directory)

