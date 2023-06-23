import os
import re

# Prompt the user for the directory path
directory = input("Enter the directory path: ")

# Prompt the user for the desired name of the resulting file
output_file_name = input("Enter the name of the resulting file: ")

# List the files in the directory
file_list = os.listdir(directory)

# Create a result list
results = []

# Process each file in the directory
for file_name in file_list:
    file_path = os.path.join(directory, file_name)
    with open(file_path, "r") as file:
        content = file.read()
        matches = re.findall(r'cluster_rank="1"([^<]+)', content)
        for match in matches:
            cleaned_match = re.sub(r'[^0-9+\-\.]+', ',', match)  # Use \. to match a literal dot
            if cleaned_match:
                result_line = f"{file_name}-{cleaned_match}"
                results.append(result_line)

# Remove the ".xml-" string and the last comma from each result
cleaned_results = [line.replace('.xml-', '')[:-1] for line in results]

# Create the header line
header_line = "Ligand,lowest_binding_energy,run,mean_binding_energy,num_in_clus"

# Write the cleaned results to the output file
with open(output_file_name, "w") as result_file:
    result_file.write(header_line + "\n")
    result_file.write("\n".join(cleaned_results))

print(f"Results saved in {output_file_name}")

