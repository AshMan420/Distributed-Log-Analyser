from mpi4py import MPI
import os
import sys
import time
from collections import defaultdict

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Same 3 functions as the sequential analyzer
def merge_counts(total_counts, new_counts):
    """
    Merge counts from one analysis into the total.

    Args:
        total_counts: Existing accumulated counts
        new_counts: New counts to add
    """
    for level, count in new_counts.items():
        total_counts[level] += count

def analyse_log_file(filepath):
    """
    Analyse a single log file and count occurrences of each log level.

    Args:
        filepath: Path to the log file

    Returns:
        Dictionary with counts for each log level (INFO, WARN, ERROR, etc.)
    """
    counts = defaultdict(int)

    try:
        with open(filepath, 'r') as f:
            for line in f:
                # Simple parsing: look for log levels in brackets
                # Example: [2025-03-27 12:02:03] [ERROR] Disk read failed
                if '[INFO]' in line:
                    counts['INFO'] += 1
                elif '[WARN]' in line or '[WARNING]' in line:
                    counts['WARN'] += 1
                elif '[ERROR]' in line:
                    counts['ERROR'] += 1
                elif '[DEBUG]' in line:
                    counts['DEBUG'] += 1
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

    return counts

def log_files(log_dir):
    files=[]
    for filename in os.listdir(log_dir):
        if filename.endswith('.log'):
            files.append(os.path.join(log_dir,filename))
    return files

# Function to divide the log files between the processes
def divide_work(files, size):
    block=len(files)//size
    rem=len(files)%size

    blocks_array = []
    count=0
    for i in range(size):
        
        num_to_take = block 
        if i < rem:            # The processes less than the remainder will get an extra file to analyse each
            num_to_take += 1

        blocks_array.append(files[count:count+num_to_take])

        count += num_to_take

        #count += block+1 if (i<rem) else block

    return blocks_array



def main():
    if rank==0:
        
        if len(sys.argv) < 2:
            print("Usage: python base_log_analyser.py <log_directory>")
            sys.exit(1)

        log_dir = sys.argv[1]
        files = log_files(log_dir)

        if not os.path.isdir(log_dir):
            print(f"Error: {log_dir} is not a valid directory")
            sys.exit(1)

        if not files:
            print(f"No .log files found in {log_dir}")
            sys.exit(1)

        files_per_process = divide_work(files,size)   # Gets the files allocated to each process

        
    else:    
        files_per_process = None   # Initialises files_per_process for each process

    start_time = time.time()
    process = comm.scatter(files_per_process, root=0)   # Finally allocates the files to analyse to each process 


    # Merging individual file counts into a single one for each process
    local_counts=defaultdict(int)
    for file in process:
        counts = analyse_log_file(file)
        merge_counts(local_counts, counts)   # Every process will have its own version of local_counts, containing the consolidated results for every file allocated to it

    result = comm.gather(local_counts, root=0)   # Gets the results from each local_counts from every process and stores in the master process

    end_time = time.time()

    if rank==0:
        total_counts = defaultdict(int)

        # Takes the list of dictionaries and stores all the results in a single total_counts dictionary
        for res in result:                  
            merge_counts(total_counts,res)

        print(f"Found {len(files)} log file(s) to analyse")
        print("Starting parallel analysis...")

        print("\n" + "="*50)
        print("ANALYSIS RESULTS")
        print("="*50)

        for level in sorted(total_counts.keys()):
            print(f"{level}: {total_counts[level]}")

        print("="*50)
    
        print(f"Total time: {end_time - start_time:.2f}s")
        
        if size > 1:
            SIZE_1_TIME = 1.39   # Found by taking the average on 5-6 independent runs
            print(f"Speedup: {SIZE_1_TIME/(end_time - start_time):.1f}x")
        print("="*50)

        


if __name__ == "__main__":
    main()
