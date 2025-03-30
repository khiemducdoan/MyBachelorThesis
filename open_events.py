import os
import argparse
import subprocess
from tensorboardX import SummaryWriter

def open_tensorboard(log_dir):
    """Launch TensorBoard using the specified log directory."""
    
    # Check if the directory exists
    if not os.path.exists(log_dir):
        print(f"Error: The specified log directory '{log_dir}' does not exist.")
        return
    
    print(f"Opening TensorBoard with logs from: {log_dir}")
    
    # Run TensorBoard
    subprocess.run(["tensorboard", "--logdir", log_dir, "--port", "6001"], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open TensorBoard with an events file")
    parser.add_argument("--log_dir", type=str, required=True, help="Path to the log directory containing event files")
    args = parser.parse_args()
    open_tensorboard(args.log_dir)
