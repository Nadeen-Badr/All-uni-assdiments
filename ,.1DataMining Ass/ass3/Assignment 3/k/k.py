def max_machines_with_budget(R, C, D, NR, NC, ND, PR, PC, PD, N):
    def can_run_machines(m):
        # Calculate additional resources needed
        additional_ram = max(0, m * R - NR)
        additional_cpu = max(0, m * C - NC)
        additional_disk = max(0, m * D - ND)
        
        # Calculate the total cost for additional resources
        total_cost = additional_ram * PR + additional_cpu * PC + additional_disk * PD
        return total_cost <= N
    
    low, high = 0, 10**9  # An upper bound for binary search
    while low < high:
        mid = (low + high + 1) // 2
        if can_run_machines(mid):
            low = mid
        else:
            high = mid - 1
    
    return low

# Get the absolute path of the current script
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the paths to the input and output files
input_file_path = os.path.join(current_dir, 'i.txt')
output_file_path = os.path.join(current_dir, 'output.txt')

# Open the input file and read the data
with open(input_file_path, 'r') as file:
    lines = file.readlines()

# First line contains the number of test cases
T = int(lines[0].strip())

results = []

# Process each test case
line_index = 1
for _ in range(T):
    R, C, D = map(int, lines[line_index].strip().split())
    NR, NC, ND = map(int, lines[line_index + 1].strip().split())
    PR, PC, PD = map(int, lines[line_index + 2].strip().split())
    N = int(lines[line_index + 3].strip())
    results.append(max_machines_with_budget(R, C, D, NR, NC, ND, PR, PC, PD, N))
    line_index += 4

# Write the results to the output file
with open(output_file_path, 'w') as file:
    for result in results:
        file.write(f"{result}\n")