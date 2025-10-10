import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

def select_file():
    # Create and hide the main tkinter window
    root = tk.Tk()
    root.withdraw()

    # Show the file dialog and get the selected file path
    file_path = filedialog.askopenfilename(
        title="Select CSV File",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialdir=os.path.expanduser("~")  # Start in the user's home directory
    )
    
    return file_path

def csv_to_jsonl(file_path):
    if not file_path:  # If user cancels file selection
        print("No file selected. Exiting...")
        return

    # Read CSV file
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        messagebox.showerror("Error", f"File {file_path} not found.")
        return
    except Exception as e:
        messagebox.showerror("Error", f"Error reading CSV file: {str(e)}")
        return

    # Drop the 'id' column if it exists
    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    # Check if required columns exist
    required_columns = {'Question', 'Answer'}
    if not required_columns.issubset(df.columns):
        messagebox.showerror("Error", "CSV file must contain 'Question' and 'Answer' columns.")
        return

    # Create the template string for each entry
    def create_jsonl_entry(row):
        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>  \n\nYou are a helpful AI assistant which provides detailed answers for RICS APC candidates relevant to their desired areas of competency. You will help RICS APC candidates formulate their submissions.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{row['Question']}<|start_header_id|>assistant<|end_header_id|>{row['Answer']}**<|eot_id|>**"
        return {"text": text}

    # Convert DataFrame to list of dictionaries
    jsonl_data = [create_jsonl_entry(row) for _, row in df.iterrows()]

    # Shuffle and split the data
    train_data, temp_data = train_test_split(jsonl_data, test_size=0.2, random_state=42)
    valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Get directory path from input file
    directory = os.path.dirname(file_path)
    if not directory:
        directory = '.'

    # Function to save JSONL file
    def save_jsonl(data, filename):
        full_path = os.path.join(directory, filename)
        with open(full_path, 'w', encoding='utf-8') as f:
            for entry in data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        return full_path

    # Save the files and get their paths
    train_path = save_jsonl(train_data, 'train.jsonl')
    valid_path = save_jsonl(valid_data, 'valid.jsonl')
    test_path = save_jsonl(test_data, 'test.jsonl')

    # Show success message with file locations
    success_message = (
        f"Files successfully created:\n\n"
        f"Training samples ({len(train_data)}): {train_path}\n"
        f"Validation samples ({len(valid_data)}): {valid_path}\n"
        f"Testing samples ({len(test_data)}): {test_path}"
    )
    messagebox.showinfo("Success", success_message)

def main():
    file_path = select_file()
    csv_to_jsonl(file_path)

if __name__ == "__main__":
    main()