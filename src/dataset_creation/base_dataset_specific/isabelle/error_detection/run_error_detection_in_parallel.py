import os
import subprocess
from pathlib import Path
import time

from src.typing import SPLIT
from src.dataset_creation.base_dataset_specific.isabelle.informal_to_formal.\
    convert_to_formal import IsabelleInformalToFormalTap
from src.dataset_creation.base_dataset_specific.isabelle.error_detection.\
    utils import kill_process_to_clean_up_isabelle
from src.dataset_creation.base_dataset_specific.isabelle.error_detection.\
    preprocessing import get_batch_file_names_path
from src.utils.os import kill_process_using_port


working_dir = os.getcwd()
sub_terminal_output_dir = Path(f"{working_dir}/terminal_outputs")


class ParallelErrorDetectionTap(IsabelleInformalToFormalTap):
    split: SPLIT
    batch_idx_start: int
    batch_idx_end: int
    start_port: int = 8000
    overwrite_results: bool = False


def main():
    args = ParallelErrorDetectionTap().parse_args()
    
    # get PISA_PATH environment variable
    pisa_path = os.environ["PISA_PATH"]
    
    # clean up sub_terminal_output_dir if exists
    if sub_terminal_output_dir.exists():
        for file in sub_terminal_output_dir.iterdir():
            file.unlink()
    sub_terminal_output_dir.mkdir(parents=True, exist_ok=True)
    
    # kill previous processes
    kill_process_to_clean_up_isabelle()
    
    # run error detection in parallel
    try:
        sub_list = []
        for batch_idx in range(args.batch_idx_start, args.batch_idx_end+1):
            port = args.start_port + batch_idx
            batch_files_list = get_batch_file_names_path(
                dataset_name=args.dataset_name,
                initial_generation_model_name=args.base_model_name,
                conversion_model_name=args.conversion_model_name,
                split=args.split, batch_idx=batch_idx
            )
            
            args_list = [
                "--split", args.split,
                "--batch_file_names_path", str(batch_files_list),
                "--port", str(port)
            ]
            if args.overwrite_results:
                args_list.append("--overwrite_results")
            
            args_text = " ".join(args_list)
            
            working_dir = os.getcwd()
            mod_id = batch_idx % 8
            isa_path = f'{working_dir}/isabelle_copy/isabelle_copy_{mod_id}/main_isa/Isabelle2022'
            
            # update enviornment variables
            env = os.environ.copy()
            env["PISA_PATH"] = pisa_path
            env["ISABELLE_HOME"] = isa_path
            # env["JAVA_TOOL_OPTIONS"] = "-Xmx512m"
            
            sub = subprocess.Popen(
                # 'ulimit -a &&' + \  # this is for debug
                f'python src/dataset_creation/base_dataset_specific/isabelle/error_detection/error_detection.py {args_text} | tee {str(sub_terminal_output_dir)}/error_detection_{args.split}_{port}.txt',
                shell=True, env=env
            )
            sub_list.append(sub)
            
            time.sleep(10)
        
        # when shell=True, sub.wait() will not work
        # we need to keep the main process running and stop it manually
        while True:
            # in every 10 minutes, print a message
            time.sleep(600)
            print(f"Running error detection for {args.split} split -- error_detection.py")
            print("If you want to stop the process, please kill the main process.")
            pass
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        for sub in sub_list:
            # this will not work when shell=True
            # sub.wait()
            # print(f"Finished process with pid {sub.pid} for {args.split} split -- error_detection.py")
            
            sub.kill()
            print(f"Killed process with pid {sub.pid} for {args.split} split -- error_detection.py")

        # make sure to kill all processes
        for batch_idx in range(args.batch_idx_start, args.batch_idx_end+1):
            kill_process_using_port(args.start_port + batch_idx)
    
        # kill previous processes
        kill_process_to_clean_up_isabelle()


if __name__ == "__main__":
    main()
