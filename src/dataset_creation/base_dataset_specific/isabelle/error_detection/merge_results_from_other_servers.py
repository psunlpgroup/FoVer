from pathlib import Path
import subprocess
import json

from tap import Tap

from src.dataset_creation.base_dataset_specific.isabelle.error_detection.\
    preprocessing import isabelle_generated_thy_files_dir
from src.dataset_creation.base_dataset_specific.isabelle.error_detection.\
    error_detection import is_isabelle_result_including_error


class MergeResultsFromOtherServersArgs(Tap):
    server_name: str
    thy_dir_path_in_another_server: str


tmp_thy_files_dir = Path("tmp_thy_files")


def main():
    args = MergeResultsFromOtherServersArgs().parse_args()
    
    if isabelle_generated_thy_files_dir.exists():
        # delete
        isabelle_generated_thy_files_dir.rmdir()
    isabelle_generated_thy_files_dir.mkdir(parents=True)
    
    # copy from another server using scp
    scp_command = f"scp -r {args.server_name}:{args.thy_dir_path_in_another_server} {tmp_thy_files_dir}"
    subprocess.run(scp_command, shell=True, check=True)
    
    # move files from tmp_thy_files_dir to isabelle_generated_thy_files_dir, but check if the files are already there
    # first, move .thy files if they are not already there
    for thy_file in tmp_thy_files_dir.glob("**/*.thy"):
        # replace parent directry with isabelle_generated_thy_files_dir
        target_file = isabelle_generated_thy_files_dir / thy_file.relative_to(tmp_thy_files_dir)
        if not target_file.exists():
            target_file.parent.mkdir(parents=True, exist_ok=True)
            thy_file.replace(target_file)
    
    # next, move .json files (results)
    for json_file in tmp_thy_files_dir.glob("**/*.json"):
        target_file = isabelle_generated_thy_files_dir / json_file.relative_to(tmp_thy_files_dir)
        
        replace = False
        if not target_file.exists():
            replace = True
        else:
            # check if the existing file includes error
            with open(target_file, "r") as f:
                target_json = json.load(f)
            
            # if empty
            if not target_json:
                replace = True
            
            # other errors
            if is_isabelle_result_including_error(target_json):
                replace = True
            
        if replace:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            json_file.replace(target_file)
    
    # remove tmp_thy_files_dir
    subprocess.run(f"rm -r {tmp_thy_files_dir}", shell=True, check=True)


if __name__ == "__main__":
    main()
