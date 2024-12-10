import getpass

from src.utils.os import kill_process_by_user_and_name


def kill_process_to_clean_up_isabelle() -> None:
    """ Kill the Isabelle processes to clean up the environment. """
    
    user_name = getpass.getuser()
    processes_to_kill = ["isabelle/error_detection/error_detection.py"]  # , "Isabelle", "poly", "sbt", "scala"]

    for process_name in processes_to_kill:
        kill_process_by_user_and_name(user_name, process_name)
    
    print("Cleaned up Isabelle processes.")
