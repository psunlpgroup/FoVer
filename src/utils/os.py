import os
import subprocess
import time


def kill_process_using_port(port):
    try:
        # Find all PIDs using the specified port
        result = subprocess.run(['lsof', '-t', f'-i:{port}'], stdout=subprocess.PIPE, text=True)
        pids = result.stdout.strip().split("\n")

        if pids and pids[0]:  # Ensure there are valid PIDs
            for pid in pids:
                print(f"Killing process {pid} using port {port}...")
                os.system(f'kill -9 {pid}')
            
            time.sleep(10)
            print("All processes using the port have been killed successfully.")
        else:
            print(f"No process found using port {port}.")
    
    except Exception as e:
        print(f"Error: {e}")


def kill_process_by_user_and_name(user, process_name):
    try:
        # Run 'ps' safely without shell=True
        result = subprocess.run(
            ["ps", "-u", user, "-o", "pid,command"], stdout=subprocess.PIPE, text=True
        )
        
        lines = result.stdout.strip().split("\n")
        pids = []

        # Skip the header and filter processes
        for line in lines[1:]:  
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and process_name in parts[1]:
                pids.append(parts[0])  # First column is the PID

        if pids:
            for pid in pids:
                subprocess.run(["kill", "-9", pid])
                print(f"Killed process {pid} ({process_name}) owned by {user}")
        else:
            print(f"No process '{process_name}' found for user '{user}'.")

    except Exception as e:
        print(f"Error: {e}")
