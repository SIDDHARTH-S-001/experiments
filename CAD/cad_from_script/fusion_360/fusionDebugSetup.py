import subprocess
import os
import sys

class FusionDebugSetup:
    def __init__(self, pre_run_path: str):
        """
        Initialize with full path to pre-run.py
        :param pre_run_path: Full path to pre-run.py
        """
        if not os.path.isfile(pre_run_path):
            raise FileNotFoundError(f"pre-run.py not found at: {pre_run_path}")
        self.pre_run_path = os.path.abspath(pre_run_path)

    def run(self):
        """
        Execute the pre-run.py script using the same Python interpreter
        as the current process.
        """
        try:
            print(f"[FusionDebugSetup] Executing: {self.pre_run_path}")
            subprocess.run([sys.executable, self.pre_run_path], check=True)
            print(f"[FusionDebugSetup] Execution completed.")
        except subprocess.CalledProcessError as e:
            print(f"[FusionDebugSetup] Script execution failed: {e}")
            raise
