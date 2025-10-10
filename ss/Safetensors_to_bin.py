import tkinter as tk
from tkinter import filedialog, messagebox
import os
from safetensors.torch import load_file
import torch

def main():
    # Initialize a hidden Tkinter window
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Open a file dialog to select the safetensors file
    file_path = filedialog.askopenfilename(
        title="Select the safetensors file",
        filetypes=[("Safetensors files", "*.safetensors"), ("All files", "*.*")]
    )

    if not file_path:
        messagebox.showinfo("No File Selected", "No file was selected. Exiting the conversion.")
        return

    # Define the output path: same directory as input, with the name 'adapter_model.bin'
    output_file = os.path.join(os.path.dirname(file_path), "adapter_model.bin")

    try:
        # Load the state dictionary from the safetensors file
        state_dict = load_file(file_path)
        # Save it as a PyTorch binary file
        torch.save(state_dict, output_file)
        messagebox.showinfo("Conversion Successful", f"File converted successfully:\n{output_file}")
    except Exception as e:
        messagebox.showerror("Conversion Failed", f"An error occurred:\n{e}")

if __name__ == "__main__":
    main()
