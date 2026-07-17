from mpi4py import MPI
import json
import os
import sys
import time
import datetime
from collections import defaultdict

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
status = MPI.Status()

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

'''
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
'''

def check_for_timeouts(tasks_in_progress, queue, dead_workers, workers_stopped, timed_out_tasks_set, TIMEOUT):
    timed_out_tasks = []
    for task, (worker_rank, start_time) in tasks_in_progress.items():
        if time.time() - start_time > TIMEOUT:
            queue.append(task)   # Adding the failed task back to the queue
            timed_out_tasks.append(task)

            timed_out_tasks_set.add(task)   # For printing which task is getting reassigned

            if worker_rank not in dead_workers:
                print(f"[Master] Worker {worker_rank} timeout detected – marking as failed")
                dead_workers.add(worker_rank)
                workers_stopped += 1

    for task in timed_out_tasks:
        tasks_in_progress.pop(task)
    
    return workers_stopped   # Need to return this because it is pass by value

def save_checkpoint(total_counts, files_processed, queue, CHECKPOINT_DIR, tasks_in_progress, dead_workers):
    timestamp = datetime.datetime.now().isoformat()
    dir_name = os.path.join(CHECKPOINT_DIR, f"checkpoint-{timestamp.replace(':','-').replace('.','-')}")   # Replaced the colons with hyphens because colons aren't allowed in directory names
    os.makedirs(dir_name, exist_ok=True)

    data = {
        "timestamp": timestamp,
        "counts": total_counts,
        "files_processed": files_processed,
        "pending_files": queue + list(tasks_in_progress.keys()),
        "failed_workers": list(dead_workers)
    }

    
    json_path = os.path.join(dir_name, "checkpoint.json")
    tmp_path = os.path.join(dir_name, "checkpoint.json.tmp")   # To prevent corruption when crash is during a save
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=4)
    os.rename(tmp_path, json_path)

def recover(CHECKPOINT_DIR):
    checkpoint_folders = os.listdir(CHECKPOINT_DIR)
    checkpoint_paths=[]

    for folder in checkpoint_folders:
         if folder.startswith("checkpoint-"):
            full_path = os.path.join(CHECKPOINT_DIR, folder)
            checkpoint_paths.append(full_path)

    
    latest_checkpoint = sorted(checkpoint_paths)[-1]
    json_path = os.path.join(latest_checkpoint, "checkpoint.json")

    with open(json_path,'r') as f:
        data = json.load(f)
    
    total_counts = defaultdict(int, data["counts"])
    files_processed = data["files_processed"]
    queue = data["pending_files"]
    dead_workers = set(data["failed_workers"])

    return total_counts, files_processed, queue, dead_workers

def main():
    # Master logic
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


        initial_time = time.time()

        # Initialising and recovery logic
        CHECKPOINT_DIR = "checkpoints"
        checkpoints_saved_count = 0
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        if os.listdir(CHECKPOINT_DIR):
            print("Loading checkpoint from previous run...")
            total_counts, files_processed, queue, dead_workers = recover(CHECKPOINT_DIR)
            tasks_in_progress={}
            workers_stopped = len(dead_workers)
            timed_out_tasks_set = set()
            print(f"Resuming from checkpoint: {len(files_processed)} files already processed")
        
        else:
            queue = files

            total_counts=defaultdict(int)
            tasks_in_progress={}
            files_processed=[]
            dead_workers = set()   # Set for tracking dead workers
            workers_stopped = 0
            timed_out_tasks_set = set()
        
        print(f"Found {len(files)} log file(s) to analyse ({len(queue)} remaining)")
        print("Starting distributed analysis...")

        workers = size - 1   # 1 master and size-1 workers
        
        
       
        
        
        # The loop runs until all work has been completed 
        while workers_stopped < workers:
            
            
            if comm.iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG):   # For fault detection
                message = comm.recv(source=MPI.ANY_SOURCE, status=status, tag=MPI.ANY_TAG)
                #sender_rank = status.Get_source()   # Instead of using status, we can have the master receieve the rank itself as the message for tag=0
            
                received_tag=status.Get_tag()

                
                
                if received_tag==0:
                    sender_rank=message   # If the tag is 0, the worker has sent its rank

                    if sender_rank in dead_workers:
                        comm.send("stop", dest=sender_rank, tag=1)   

                    if queue:
                        task=queue.pop()
                        comm.send(task,dest=sender_rank,tag=1)
                        tasks_in_progress[task] = (sender_rank, time.time())

                    # This elif and else block is for handling the case when all workers have been stopped and THEN a worker gets timed out
                    # It asks the healthy workers to wait before stopping in order to complete the last file
                    elif not queue and not tasks_in_progress:
                        comm.send("stop",dest=sender_rank,tag=1)
                        workers_stopped+=1
                    # Runs when queue is empty AND tasks_in_progress is not empty
                    else:
                        comm.send("wait", dest=sender_rank, tag=1)


                        # Checking if the task is being assigned or reassigned
                        if task in timed_out_tasks_set:
                            print(f"[Master] Reassigning {task} to worker {sender_rank}")
                            timed_out_tasks_set.remove(task)
                        else:
                            print(f"[Master] Assigning {task} to worker {sender_rank}")

                    

                elif received_tag==2:
                    # We're ignoring the result sent from a possible slow worker taking > 3s, that got stopped but is still processing the file it got sent last
                    if sender_rank not in dead_workers:
                        task, result = message   # If the tag is 2, the worker has sent the analysed result from the file and the filename
                        merge_counts(total_counts, result)

                        files_processed.append(task)
                        tasks_in_progress.pop(task)   # Remove the completed task from tasks_in_progress

                        # Checkpointing logic
                        if(len(files_processed)%100==0):
                            save_checkpoint(total_counts, files_processed, queue, CHECKPOINT_DIR, tasks_in_progress, dead_workers)
                            checkpoints_saved_count += 1
            

            # Checking for failed workers/tasks
            TIMEOUT = 3.0 
            workers_stopped = check_for_timeouts(tasks_in_progress, queue, dead_workers, workers_stopped, timed_out_tasks_set, TIMEOUT)
                

            

        total_time = time.time() - initial_time
        print("\n========================================")
        print(" ANALYSIS RESULTS")
        print("========================================")
        for level, count in total_counts.items():
            print(f"– {level}: {count}")
        print("========================================")
        print(f"Total time: {total_time:.2f}s")
        print(f"Files processed: {len(files_processed)}")
        print(f"Failed workers: {len(dead_workers)}")
        print(f"Checkpoints saved: {checkpoints_saved_count}")
        print("========================================")
    



    # Worker logic  
    else:    
        comm.send(rank,dest=0,tag=0)
        
        while True:
            work = comm.recv(source=0, tag=1)   # Work received from master

            # If there is no task left to assign, or the worker is declared dead, the worker is stopped
            if work=="stop":
                break

            if work == "wait":
                comm.send(rank, dest=0, tag=0)
                continue # Go back to the top of the loop
            
            '''
            if rank == 2:
        
                time.sleep(5)  # Work for a bit
                print(f"[Rank {rank}] Simulating crash...")
                
                sys.exit(1)  # Crash!
            '''

            print(f"[Worker {rank}] Processing {work}")
            result = analyse_log_file(work)
            comm.send((work,result), dest=0, tag=2)   # Send back the analysed file as well as the filename, so tasks_in_progress can be updated
            comm.send(rank, dest=0, tag=0)     # Ask for more work when done. This can also be avoided if you send back work when tag=2 is received in master as well

    

   
    

    
if __name__ == "__main__":
    main()


'''
# Abandoned logic
rank = comm.recv(source=MPI.ANY_SOURCE, status=status, tag=0) 
task = queue.pop()
comm.send(task, dest=req, tag=1)
record[task] = req
result = comm.recv(source=MPI.ANY_SOURCE, tag=2)
'''
