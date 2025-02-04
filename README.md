# Fine-tuning Large Language Models on Apple Silicon Macs using MLX & Llama.cpp
This repository aims at providing a detailed guidance about how you can fine-tune large language models on Apple Silicon Macs using MLX and llamac.cpp.

## Acknowledgements
A number of resources helped in creating this guidance. This repository aims at collating the most useful information from all the following resources:
* [The MLX community](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md)
* Andy Peating articles:
    * [Part 1 - Setting up your environment](https://apeatling.com/articles/part-1-setting-up-your-environment/)
    * [Part 2 - Building your training data for fine-tuning](https://apeatling.com/articles/part-2-building-your-training-data-for-fine-tuning/)
    * [Part 3 - Fine-tuning your llm using the mlx framework](https://apeatling.com/articles/part-3-fine-tuning-your-llm-using-the-mlx-framework/)
    * [Part 4- Testing and interacting with your fine-tuned LLM](https://apeatling.com/articles/part-4-testing-and-interacting-with-your-fine-tuned-llm/)
* [Llama 3 Model cards and prompting format available through](https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/)
* [Fine-tuning LLMs on Mac OS using MLX and run with Ollama](https://medium.com/rahasak/fine-tuning-llms-on-macos-using-mlx-and-run-with-ollama-182a20f1fd2c)

Please note that different models have different model cards and prompting templates and you should visit the developer of every LLM to tweak the code to account for the relevant chat template.

I'll fix the formatting and links in your table of contents. Here's the corrected version:

## Table of Contents
* [Part I - Setting the Coding Environment](#part-i---setting-the-coding-environment)
* [Part II - Create/Import data for Finetuning](#part-ii---createimport-data-for-finetuning)
* [Part III - Training the model, testing and validation](#part-iii---training-the-model-testing-and-validation)
* [Part IV - Saving the fused model with the trained adapters & compression to GGUF format](#part-iv---saving-the-fused-model-with-the-trained-adapters--compression-to-gguf-format)

## Part I - Setting the Coding Environment
### Step 1.1:
Open terminal in your preferred location on your Mac. You can do that by opening finder and then if you have the path bar enabled, you can go to your preferred location, right click on the path bar and then choose open in terminal.

### Step 1.2:
Install home-brew from <https://brew.sh/> where you have to paste the following code in terminal 

             /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

### Step 1.3:
Make sure that your home-brew is on your PATH by typing on the following commands in your opened terminal window:
            
            echo >> /Users/{Your_Username}/.zprofile
            
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /Users/{Your_Username}/.zprofile
            
            eval "$(/opt/homebrew/bin/brew shellenv)"

### Step 1.4:
Install **git** in your terminal window using the following code. 

            brew install git

   You can have more details from the git website: <https://git-scm.com/downloads/mac>

### Step 1.5:
Clone the mlx repository through the following code in your terminal window:
                 
            git clone https://github.com/ml-explore/mlx-examples.git

### Step 1.6:
Change the directory to go inside the lora folder which is located inside the mlx-examples folder through the following command in your terminal window:
     
            cd mlx-examples/lora

### Step 1.7:
Make sure to have python installed on your machine. You can go to the website <https://www.python.org/downloads/macos/> and download your preferred version. I would recommend a version that is 3.11 and later.

### Step 1.8:
Make sure to have pip installed on your machine. If in doubt you can install it using the following command in your terminal

             Python -m ensurepip —upgrade

             OR

             Python3 -m ensurepip —upgrade

### Step 1.9:
Following from **Step 1.5**, having changed the directory to be inside the lora folder, we can now Install the requirements  by typing the following command in your terminal 

            Pip install -r requirements.txt
            
            OR
            
            pip3 install -r requirements.txt

### Step 1.10:
The **MLX LoRA fine-tuning** is quite efficient for accounting for the required data, fine-tuning the model and fusing the original model with the trained adapters.
    
However when it comes to converting the safetensors of the fine-tuned model to **GGUF** for further usage on **Ollama** or **Open WebUI** or other 3rd party apps, it is mandatory to use the llama.cpp

The **mlx_lm.convert** doesnot provide multiple options for quantization, the default is **16bit**. The information in this repository is based on  February 2025 updates. This can change in future releases.

You can clone the **llama.cpp** respository which can be through the following link. The code is as follows:

            git clone https://github.com/ggerganov/llama.cpp.git

Then later on in this code, we would need to change the working directory to be inside the cloned llama.cpp on our machine to be able to use the convert_hf_to_gguf.py file for model conversion from safetensors to GGUF with the required quantization.
## Part II - Create/Import data for Finetuning
## Part III - Training the model, testing and validation
## Part IV - Saving the fused model with the trained adapters & compression to GGUF format
