import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple
import random

class ChatTemplateConverter:
    """Main application for converting datasets to LLM chat templates"""
    
    # Comprehensive chat templates for various LLM architectures
    CHAT_TEMPLATES = {
        "Llama 2": {
            "description": "Meta's Llama 2 Chat format",
            "template": "[INST] {question} [/INST] {answer}"
        },
        "Llama 3": {
            "description": "Meta's Llama 3 Chat format",
            "template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"
        },
        "Mistral/Mixtral": {
            "description": "Mistral AI's instruction format",
            "template": "[INST] {question} [/INST] {answer}"
        },
        "ChatML (OpenAI)": {
            "description": "ChatML format used by GPT models",
            "template": "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
        },
        "Alpaca": {
            "description": "Stanford Alpaca instruction format",
            "template": "### Instruction:\n{question}\n\n### Response:\n{answer}"
        },
        "Vicuna": {
            "description": "Vicuna chat format",
            "template": "USER: {question}\nASSISTANT: {answer}"
        },
        "Zephyr": {
            "description": "HuggingFace Zephyr format",
            "template": "<|user|>\n{question}</s>\n<|assistant|>\n{answer}</s>"
        },
        "Phi": {
            "description": "Microsoft Phi models format",
            "template": "Instruct: {question}\nOutput: {answer}"
        },
        "Gemma": {
            "description": "Google Gemma format",
            "template": "<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n{answer}<end_of_turn>"
        },
        "ChatGLM": {
            "description": "ChatGLM format",
            "template": "[Round 1]\n\n问：{question}\n\n答：{answer}"
        },
        "Qwen": {
            "description": "Alibaba Qwen format",
            "template": "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
        },
        "Falcon": {
            "description": "TII Falcon format",
            "template": "User: {question}\nAssistant: {answer}"
        },
        "MPT": {
            "description": "MosaicML MPT format",
            "template": "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
        },
        "OpenChat": {
            "description": "OpenChat format",
            "template": "GPT4 Correct User: {question}<|end_of_turn|>GPT4 Correct Assistant: {answer}<|end_of_turn|>"
        },
        "Custom": {
            "description": "Define your own template",
            "template": "{question}\n{answer}"
        }
    }
    
    def __init__(self, root):
        self.root = root
        self.root.title("LLM Chat Template Converter")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Data storage
        self.df = None
        self.file_path = None
        self.question_col = None
        self.answer_col = None
        self.custom_template = "{question}\n{answer}"
        
        self.setup_ui()
    
    def setup_ui(self):
        """Create the user interface"""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        style.configure('Section.TLabel', font=('Helvetica', 11, 'bold'))
        style.configure('Action.TButton', font=('Helvetica', 10), padding=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="🤖 LLM Chat Template Converter", 
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Left Panel - File and Column Selection
        left_panel = ttk.LabelFrame(main_frame, text="1. Data Source", padding="15")
        left_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # File selection
        ttk.Label(left_panel, text="Input File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.file_label = ttk.Label(left_panel, text="No file selected", foreground="gray")
        self.file_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        ttk.Button(left_panel, text="📁 Browse File", command=self.load_file,
                  style='Action.TButton').grid(row=2, column=0, columnspan=2, pady=10, sticky=tk.EW)
        
        # Column selection
        ttk.Label(left_panel, text="Question Column:", style='Section.TLabel').grid(
            row=3, column=0, sticky=tk.W, pady=(15, 5))
        self.question_combo = ttk.Combobox(left_panel, state='disabled', width=25)
        self.question_combo.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=5)
        
        ttk.Label(left_panel, text="Answer Column:", style='Section.TLabel').grid(
            row=5, column=0, sticky=tk.W, pady=(10, 5))
        self.answer_combo = ttk.Combobox(left_panel, state='disabled', width=25)
        self.answer_combo.grid(row=6, column=0, columnspan=2, sticky=tk.EW, pady=5)
        
        ttk.Button(left_panel, text="✓ Confirm Columns", command=self.confirm_columns,
                  style='Action.TButton').grid(row=7, column=0, columnspan=2, pady=15, sticky=tk.EW)
        
        # Dataset split configuration
        split_frame = ttk.LabelFrame(left_panel, text="2. Dataset Split", padding="10")
        split_frame.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Train split
        ttk.Label(split_frame, text="Train %:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.train_var = tk.IntVar(value=80)
        self.train_scale = ttk.Scale(split_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                    variable=self.train_var, command=self.update_split_labels)
        self.train_scale.grid(row=0, column=1, sticky=tk.EW, padx=5)
        self.train_label = ttk.Label(split_frame, text="80%")
        self.train_label.grid(row=0, column=2, padx=5)
        
        # Test split
        ttk.Label(split_frame, text="Test %:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.test_var = tk.IntVar(value=10)
        self.test_scale = ttk.Scale(split_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                   variable=self.test_var, command=self.update_split_labels)
        self.test_scale.grid(row=1, column=1, sticky=tk.EW, padx=5)
        self.test_label = ttk.Label(split_frame, text="10%")
        self.test_label.grid(row=1, column=2, padx=5)
        
        # Validation split
        ttk.Label(split_frame, text="Validation %:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.val_var = tk.IntVar(value=10)
        self.val_scale = ttk.Scale(split_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                  variable=self.val_var, command=self.update_split_labels)
        self.val_scale.grid(row=2, column=1, sticky=tk.EW, padx=5)
        self.val_label = ttk.Label(split_frame, text="10%")
        self.val_label.grid(row=2, column=2, padx=5)
        
        split_frame.columnconfigure(1, weight=1)
        
        # Template selection
        template_frame = ttk.LabelFrame(left_panel, text="3. Chat Template", padding="10")
        template_frame.grid(row=9, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Label(template_frame, text="Model Format:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.template_var = tk.StringVar(value="Llama 3")
        self.template_combo = ttk.Combobox(template_frame, textvariable=self.template_var,
                                          values=list(self.CHAT_TEMPLATES.keys()),
                                          state='readonly', width=25)
        self.template_combo.grid(row=1, column=0, sticky=tk.EW, pady=5)
        self.template_combo.bind('<<ComboboxSelected>>', self.on_template_change)
        
        self.template_desc = ttk.Label(template_frame, text="Meta's Llama 3 Chat format",
                                      foreground="blue", wraplength=250)
        self.template_desc.grid(row=2, column=0, sticky=tk.W, pady=5)
        
        ttk.Button(template_frame, text="✏️ Edit Custom Template", 
                  command=self.edit_custom_template).grid(row=3, column=0, pady=10, sticky=tk.EW)
        
        template_frame.columnconfigure(0, weight=1)
        
        # Middle Panel - Preview Area
        middle_panel = ttk.Frame(main_frame)
        middle_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Notebook for different previews
        self.notebook = ttk.Notebook(middle_panel)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Original Data Preview
        original_frame = ttk.Frame(self.notebook)
        self.notebook.add(original_frame, text="📊 Original Data")
        
        self.original_text = scrolledtext.ScrolledText(original_frame, wrap=tk.WORD,
                                                       width=60, height=30, font=('Consolas', 9))
        self.original_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Template Preview
        template_preview_frame = ttk.Frame(self.notebook)
        self.notebook.add(template_preview_frame, text="📝 Template Format")
        
        self.template_text = scrolledtext.ScrolledText(template_preview_frame, wrap=tk.WORD,
                                                       width=60, height=30, font=('Consolas', 9))
        self.template_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Applied Template Preview
        applied_frame = ttk.Frame(self.notebook)
        self.notebook.add(applied_frame, text="✨ Applied Template")
        
        self.applied_text = scrolledtext.ScrolledText(applied_frame, wrap=tk.WORD,
                                                      width=60, height=30, font=('Consolas', 9))
        self.applied_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        middle_panel.rowconfigure(0, weight=1)
        middle_panel.columnconfigure(0, weight=1)
        
        # Right Panel - Actions and Info
        right_panel = ttk.LabelFrame(main_frame, text="4. Export", padding="15")
        right_panel.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        # Dataset info
        info_frame = ttk.LabelFrame(right_panel, text="Dataset Info", padding="10")
        info_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        self.info_text = tk.Text(info_frame, height=8, width=35, font=('Consolas', 9),
                                state='disabled', bg='#f9f9f9')
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Export button
        ttk.Button(right_panel, text="🚀 Generate Training Files",
                  command=self.export_files, style='Action.TButton').grid(
                      row=1, column=0, pady=10, sticky=tk.EW)
        
        # Status frame
        status_frame = ttk.LabelFrame(right_panel, text="Status", padding="10")
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(15, 0))
        
        self.status_text = scrolledtext.ScrolledText(status_frame, height=15, width=35,
                                                     font=('Consolas', 9), state='disabled',
                                                     bg='#f9f9f9')
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
        right_panel.rowconfigure(2, weight=1)
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=3)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Initialize previews
        self.update_template_preview()
        self.log_status("Welcome! Load a CSV or Excel file to begin.")
    
    def log_status(self, message: str):
        """Add a status message"""
        self.status_text.config(state='normal')
        self.status_text.insert(tk.END, f"• {message}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state='disabled')
    
    def load_file(self):
        """Load CSV or Excel file"""
        file_path = filedialog.askopenfilename(
            title="Select Dataset File",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path)
            else:
                self.df = pd.read_excel(file_path)
            
            self.file_path = file_path
            self.file_label.config(text=Path(file_path).name, foreground="green")
            
            # Populate column dropdowns
            columns = list(self.df.columns)
            self.question_combo.config(values=columns, state='readonly')
            self.answer_combo.config(values=columns, state='readonly')
            
            # Auto-select if common column names found
            for col in columns:
                if 'question' in col.lower() or 'input' in col.lower() or 'prompt' in col.lower():
                    self.question_combo.set(col)
                if 'answer' in col.lower() or 'output' in col.lower() or 'response' in col.lower():
                    self.answer_combo.set(col)
            
            self.log_status(f"✓ Loaded {len(self.df)} rows from {Path(file_path).name}")
            self.update_info()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
            self.log_status(f"✗ Error loading file: {str(e)}")
    
    def confirm_columns(self):
        """Confirm column selection and update previews"""
        if not self.question_combo.get() or not self.answer_combo.get():
            messagebox.showwarning("Warning", "Please select both question and answer columns")
            return
        
        self.question_col = self.question_combo.get()
        self.answer_col = self.answer_combo.get()
        
        self.log_status(f"✓ Columns confirmed: Q='{self.question_col}', A='{self.answer_col}'")
        self.update_original_preview()
        self.update_applied_preview()
        self.update_info()
    
    def update_original_preview(self):
        """Show original data preview"""
        if self.df is None or self.question_col is None:
            return
        
        self.original_text.delete(1.0, tk.END)
        preview = self.df[[self.question_col, self.answer_col]].head(5)
        self.original_text.insert(tk.END, "First 5 rows of your dataset:\n\n")
        self.original_text.insert(tk.END, preview.to_string(index=False))
    
    def update_template_preview(self):
        """Show template format preview"""
        template_name = self.template_var.get()
        if template_name == "Custom":
            template = self.custom_template
        else:
            template = self.CHAT_TEMPLATES[template_name]["template"]
        
        self.template_text.delete(1.0, tk.END)
        self.template_text.insert(tk.END, "Template Structure:\n")
        self.template_text.insert(tk.END, "=" * 50 + "\n\n")
        self.template_text.insert(tk.END, template)
        self.template_text.insert(tk.END, "\n\n" + "=" * 50 + "\n\n")
        self.template_text.insert(tk.END, "Placeholders:\n")
        self.template_text.insert(tk.END, "• {question} - Will be replaced with question text\n")
        self.template_text.insert(tk.END, "• {answer} - Will be replaced with answer text\n")
    
    def update_applied_preview(self):
        """Show applied template preview with sample data"""
        if self.df is None or self.question_col is None:
            return
        
        template_name = self.template_var.get()
        if template_name == "Custom":
            template = self.custom_template
        else:
            template = self.CHAT_TEMPLATES[template_name]["template"]
        
        self.applied_text.delete(1.0, tk.END)
        self.applied_text.insert(tk.END, "Sample Formatted Examples:\n")
        self.applied_text.insert(tk.END, "=" * 50 + "\n\n")
        
        # Show 3 examples
        for idx in range(min(3, len(self.df))):
            row = self.df.iloc[idx]
            formatted = template.format(
                question=row[self.question_col],
                answer=row[self.answer_col]
            )
            self.applied_text.insert(tk.END, f"Example {idx + 1}:\n")
            self.applied_text.insert(tk.END, formatted)
            self.applied_text.insert(tk.END, "\n\n" + "-" * 50 + "\n\n")
    
    def update_split_labels(self, *args):
        """Update split percentage labels"""
        self.train_label.config(text=f"{self.train_var.get()}%")
        self.test_label.config(text=f"{self.test_var.get()}%")
        self.val_label.config(text=f"{self.val_var.get()}%")
        self.update_info()
    
    def update_info(self):
        """Update dataset info display"""
        self.info_text.config(state='normal')
        self.info_text.delete(1.0, tk.END)
        
        if self.df is not None:
            total = len(self.df)
            train_pct = self.train_var.get() / 100
            test_pct = self.test_var.get() / 100
            val_pct = self.val_var.get() / 100
            
            train_size = int(total * train_pct)
            test_size = int(total * test_pct)
            val_size = total - train_size - test_size
            
            info = f"Total Rows: {total}\n\n"
            info += f"Train: {train_size} rows\n"
            info += f"Test: {test_size} rows\n"
            info += f"Validation: {val_size} rows\n\n"
            info += f"Template: {self.template_var.get()}\n"
            
            self.info_text.insert(tk.END, info)
        
        self.info_text.config(state='disabled')
    
    def on_template_change(self, event=None):
        """Handle template selection change"""
        template_name = self.template_var.get()
        self.template_desc.config(text=self.CHAT_TEMPLATES[template_name]["description"])
        self.update_template_preview()
        self.update_applied_preview()
        self.log_status(f"Template changed to: {template_name}")
    
    def edit_custom_template(self):
        """Open dialog to edit custom template"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Custom Template")
        dialog.geometry("600x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Define your custom template:", font=('Helvetica', 11, 'bold')).pack(pady=10)
        ttk.Label(dialog, text="Use {question} and {answer} as placeholders").pack()
        
        text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, width=70, height=15,
                                        font=('Consolas', 10))
        text.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        text.insert(tk.END, self.custom_template)
        
        def save_template():
            self.custom_template = text.get(1.0, tk.END).strip()
            self.CHAT_TEMPLATES["Custom"]["template"] = self.custom_template
            self.log_status("✓ Custom template saved")
            if self.template_var.get() == "Custom":
                self.update_template_preview()
                self.update_applied_preview()
            dialog.destroy()
        
        ttk.Button(dialog, text="Save Template", command=save_template,
                  style='Action.TButton').pack(pady=10)
    
    def export_files(self):
        """Export train, test, and validation JSONL files"""
        if self.df is None or self.question_col is None:
            messagebox.showwarning("Warning", "Please load a file and confirm columns first")
            return
        
        # Validate split percentages
        total_pct = self.train_var.get() + self.test_var.get() + self.val_var.get()
        if total_pct != 100:
            messagebox.showwarning("Warning", f"Split percentages must sum to 100% (currently {total_pct}%)")
            return
        
        # Select output directory
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return
        
        try:
            # Get template
            template_name = self.template_var.get()
            if template_name == "Custom":
                template = self.custom_template
            else:
                template = self.CHAT_TEMPLATES[template_name]["template"]
            
            # Shuffle dataset
            df_shuffled = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Calculate split sizes
            total = len(df_shuffled)
            train_size = int(total * self.train_var.get() / 100)
            test_size = int(total * self.test_var.get() / 100)
            
            # Split data
            train_df = df_shuffled[:train_size]
            test_df = df_shuffled[train_size:train_size + test_size]
            val_df = df_shuffled[train_size + test_size:]
            
            # Export function
            def export_jsonl(df, filename):
                output_path = Path(output_dir) / filename
                with open(output_path, 'w', encoding='utf-8') as f:
                    for _, row in df.iterrows():
                        formatted_text = template.format(
                            question=row[self.question_col],
                            answer=row[self.answer_col]
                        )
                        json_obj = {
                            "text": formatted_text,
                            "question": row[self.question_col],
                            "answer": row[self.answer_col]
                        }
                        f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                return output_path
            
            # Export all files
            train_path = export_jsonl(train_df, "train.jsonl")
            test_path = export_jsonl(test_df, "test.jsonl")
            val_path = export_jsonl(val_df, "valid.jsonl")
            
            self.log_status(f"✓ Exported {len(train_df)} training examples to train.jsonl")
            self.log_status(f"✓ Exported {len(test_df)} test examples to test.jsonl")
            self.log_status(f"✓ Exported {len(val_df)} validation examples to valid.jsonl")
            self.log_status(f"✓ All files saved to: {output_dir}")
            
            messagebox.showinfo("Success", 
                              f"Successfully exported:\n\n"
                              f"• train.jsonl ({len(train_df)} rows)\n"
                              f"• test.jsonl ({len(test_df)} rows)\n"
                              f"• valid.jsonl ({len(val_df)} rows)\n\n"
                              f"Location: {output_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export files:\n{str(e)}")
            self.log_status(f"✗ Export failed: {str(e)}")


def main():
    root = tk.Tk()
    app = ChatTemplateConverter(root)
    root.mainloop()


if __name__ == "__main__":
    main()