# this code is partially based on dtv
# https://github.com/jinpz/dtv/blob/05f539e374096006c8618806d5ea26c24ba3f4e5/proof_checking_local.py

import os
import subprocess
import json
from pathlib import Path
from tqdm import tqdm
import time
import psutil
import signal

from tap import Tap

from src.utils.os import kill_process_using_port
from src.dataset_creation.base_dataset_specific.isabelle.\
    error_detection.preprocessing import clean_up_isabelle_statement_and_proof
from src.dataset_creation.base_dataset_specific.isabelle.\
    error_detection.run_error_detection_in_parallel import sub_terminal_output_dir
from src.dataset_creation.base_dataset_specific.isabelle.proof_checker.proof_checker import get_proof_checker


class IsabelleErrorDetectionTap(Tap):
    split: str
    batch_file_names_path: str
    port: int
    overwrite_results: bool = False


def exit_pisa_env(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for process in children:
            process.send_signal(signal.SIGTERM)
        parent.send_signal(signal.SIGTERM)
    except psutil.NoSuchProcess:
        pass


def is_isabelle_result_including_error(result: dict) -> bool:
    invalid_error_messages_list = [
        "Isabelle process has been destroyed",
        "Isabelle process was destroyed",
    ]
    
    for error_message in invalid_error_messages_list:
        if error_message in result["reason"]:
            return True
    
    return False


def main():
    args = IsabelleErrorDetectionTap().parse_args()
    
    # clean up the process using the port
    kill_process_using_port(args.port)
    
    # load file path
    with open(args.batch_file_names_path, "r") as f:
        batch_file_names: list[str] = json.load(f)
    
    # load proofs
    proofs = []
    for file_name in batch_file_names:
        with open(file_name, "r") as f:
            proofs.append("".join(f.readlines()))
    
    # check number of isabelle
    working_dir = os.getcwd()
    num_isabelle = 0
    for idx in range(100):
        candidate_dir = Path(f'{working_dir}/isabelle_copy/isabelle_copy_{idx}')
        if not candidate_dir.exists():
            num_isabelle = idx
            break
    
    mod_id = (args.port % 8000) % num_isabelle
    
    isa_path = f'{working_dir}/isabelle_copy/isabelle_copy_{mod_id}/main_isa/Isabelle2022'

    # overwrite PATH environment variable
    os.environ["PATH"] = f"{isa_path}/bin:{os.environ['PATH']}"
    
    print("which isabelle")
    subprocess.call("which isabelle", shell=True)
    
    sub = None
    terminal_file = f"{str(sub_terminal_output_dir)}/sbt_ready_{args.split}_{args.port}.txt"
    
    # run error detection
    try:
        # update environment variables
        env = os.environ.copy()
        env["ISABELLE_HOME"] = isa_path
        
        # start pisa server
        sub = subprocess.Popen(
            f"cd Portal-to-ISAbelle && echo $PATH && which isabelle && java -cp ../pisa_copy/pisa_copy{mod_id}.jar pisa.server.PisaOneStageServer{args.port} | tee {terminal_file}",
            # f'cd Portal-to-ISAbelle && echo $PATH && which isabelle && sbt "runMain pisa.server.PisaOneStageServer{args.port}" | tee {terminal_file}',
            shell=True, env=env
        )
        
        # wait for server to be activated
        environment_success = False
        failure_counter = 0
        while not environment_success and failure_counter <= 10:
            print('starting the server')
            
            sbt_ready = False
            while not sbt_ready:
                if Path(terminal_file).exists():
                    with open(terminal_file, 'r') as f:
                        file_content = f.read()
                    if 'Server is running. Press Ctrl-C to stop.' in file_content and 'error' not in file_content:
                        print('sbt should be ready')
                        sbt_ready = True
            time.sleep(3)
            
            # get proof checker
            try:
                checker = get_proof_checker(port=args.port, mod_id=mod_id)
                print('escaping the while loop')
                environment_success = True
            except:
                print('restarting the while loop')
                failure_counter += 1
                exit_pisa_env(sub.pid)
                sub.wait()
                sub.kill()
        
        # check if the environment is ready
        if not environment_success:
            raise Exception("The environment is not ready.")

        # run error detection
        for proof_idx, proof in tqdm(enumerate(proofs), total=len(proofs)):
            proof_file = Path(batch_file_names[proof_idx])
            
            # check if all_sorry.result.json exists
            sorry_result_file = proof_file.with_name("all_sorry.result.json")
            if sorry_result_file.exists():
                with open(sorry_result_file, "r") as f:
                    result = json.load(f)
                # if all sorry case does not work, there is syntax error
                # so skip the proof
                if not result["success"]:
                    print(f"Syntax error detected in {sorry_result_file}, skipped: {proof_file}")
                    continue
            
            # check if all_sledgehammer.result.json exists
            sledgehammer_result_file = proof_file.with_name("all_sledgehammer.result.json")
            if sledgehammer_result_file.exists():
                with open(sledgehammer_result_file, "r") as f:
                    result = json.load(f)
                # if all sledgehammer case does not work, there is syntax error
                # so skip the proof
                if result["success"]:
                    print(f"All steps are correct in {sledgehammer_result_file}, skipped: {proof_file}")
                    continue
            
            # get result path
            result_path = proof_file.with_suffix(".result.json")
            
            # check if the result already exists
            if result_path.exists() and not args.overwrite_results:
                with open(result_path, "r") as f:
                    result = json.load(f)
                
                if is_isabelle_result_including_error(result):
                    # re-run
                    pass
                else:
                    print(f"Already finished error detection: {result_path}")
                    print("Skipped...")
                    continue
            
            # check error
            success = False
            for attempt in range(10):
                try:
                    result = checker.check(proof)
                    success = True
                except:
                    print(f"Error in error detection: {result_path}")
                    print("Retrying...")
                    time.sleep(3)
            
            if not success:
                print(f"Error in error detection: {result_path}")
                continue
            
            with open(result_path, "w") as f:
                json.dump(result, f, indent=4)
            print(f"Finished error detection: {result_path}")
        
        print(f"Finished error detection: port={args.port}")
    
    except Exception as e:
        print(f"Error in error detection: port={args.port}")
        print(e)
        import traceback
        traceback.print_exc()
    
    finally:
        if sub is not None:
            exit_pisa_env(sub.pid)
            sub.wait()
            print(f"Finished process with pid {sub.pid} -- pisa server")
            
            sub.kill()
            print(f"Killed process with pid {sub.pid} -- pisa server")
        
        # clean up the process using the port
        kill_process_using_port(args.port)


if __name__ == "__main__":
    main()
