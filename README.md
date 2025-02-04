# Fine-tuning Large Language Models on Apple Silicon Macs using MLX & Llama.cpp
This repository aims at providing a detailed guidance about how you can fine-tune large language models on Apple Silicon Macs using the **MLX Framework** and **llama.cpp**.

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

## Repository main files:
The explanation of everything in this repository relies on the following 3 files:
* **Readme.md** - This file explains the repository, the components and the way to undertake the fine-tuning of any desired large language model using  the MLX framework. This readme file also explains how to create the data required for fine-tuning.
* **Convert_CSV_to_JSONL.py** - This python file enables the conversion of csv training dataset into 3 files **train**, **test** & **valid**.
* **Mistral_Instruct_MLX_Fine-tuning.ipynb** - This is the main jupyter notebook which contains everything related to the fine-tuning of the desired model, testing, inferencing and conversion to GGUF.

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
### Step 2.1:
This is going to be done outside of the terminal environment. For the purpose of this example, I have a **csv** file containing the questions and answers of **RICS APC** submissions in the form of questions and answers. The file is composed of 3 columns **ID**, **Question** & **Answer**. 
    
### Step 2.2:
Every model has its chat template, for the current example, we are going to use the mistral  architecture. I used a chat template that was proposed by the MLX team on their github repository which works fine for the mistral architecture
        
        {"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello."}, {"role": "assistant", "content": "How can I assistant you today."}]}

In future fine-tuning I will use the official chat template of mistral which you can find below:

         <s>[INST] Human message [/INST] Assistant response </s>[INST] Human message [/INST] Assistant response
    
### Step 2.3:
For fine-tuning purposes using the **MLX library**, we have to convert the training data into 3 json files: **train.jsonl**, **test.jsonl** & **valid.jsonl**.
    
I built a python code which takes the **CSV** uploaded by the end user from any desired location and then it applies an example chat template on it (refer to **Step 2.2** above) and then saves the training, testing and validation files in **jsonl** format. 
    
The training data is 80% of the original data size, testing is at 10% and validation is at 10%. All the resultant files are saved in a folder called **data**.

The file that undertakes the above task is **csv_to_jsonl.py** and it is available within the repository.
        
In case of gated models, you will have to install huggingface hub in terminal using:          
            
                !Pip install huggingface_hub               
            
                OR              
            
                !Pip3 install huggingface_hub 
                
This should be followed by logging into your account using your token by typing the following code in your terminal window
    
                !Huggingface-cli login —token {Your_token} 

## Part III - Training the model, testing and validation
### Step 3.1: Defining Important Variables
In this step, I am defining the different variables that I am go to use through my code.You can change them to suit your needs.

The variables include the following:
1. **Data directory** - The location where the train.jsonl, the test.jsonl and the valid.jsonl files are saved
2. **Downloaded Huggingface Model directory** - In my case it is the Mistal Instruct v0.3 with 16 bits accuracy
3. **Huggingface Repository** - In my case it is the Mistal Instruct v0.3 with 16 bits accuracy
4. My desired **Huggingface Repository Name** for saving the fine-tuned model
5. A **Write-Token** created from the Huggingface website to able to interact with the site to upload and download models.
6. A directory for the place where I want to save the converted model from the huggingface format to MLX format
7. The **Desired Name** for the fine-tuned MLX model
8. The **Output Directory** for the fine-tuned MLX model
9. **Llama.cpp Directory** where I cloned the github repository to be able to use it in terminal under this jupyter notebook.

### Step 3.2: Download the desired model from the Huggingface website
This step is quite substantial due to the importance of downloading a model from the Huggingface Website. In this project, I downloaded **Mistral Instruct v0.3 7B** with **16-bits** accuracy. The overall size of the model is circa **15GB**.

In order to download your desired model, you can use the following code inside your terminal window:
     
     !Huggingface-cli login --token {hf_token}

     !huggingface-cli download --repo-type model --local-dir {downloaded_hf_model} {hf_model}

### Step 3.3: Huggingface Model Conversion to MLX format
This step is all about converting the downloaded Huggingface model into MLX format. For this we can use the **mlx_lm.convert** function within the MLX library. The conversion for of the model could be done using the following code.

    !mlx_lm.convert \

       --hf-path {hf_model} \

       --mlx-path {mlx_path}

### Step 3.4: Undertaking the fine-tuning of the chosen model using the Low Rank Adaptation under MLX_LM

In this step, I am trying to make the best use of the Low Rank Adaptation with mlx_lm.lora under the mlx_lm package to fine-tune the already converted MLX Mistral V0.3. The summary of the fine-tuning code is as shown below. Please refer to the jupyter notebook which includes the full explanation of the arguments of the fine-tuning. 

    ! mlx_lm.lora \
     --model ${mlx_path} \
     --train \
     --data ${data} \
     --fine-tune-type lora \
     --num-layers 16 \
     --batch-size 8 \
     --iters 1000 \
     --val-batches 50 \
     --learning-rate 1e-5 \
     --steps-per-report 10 \
     --steps-per-eval 200 \
     --adapter-path ${adapters} \
     --save-every 500 \
     --max-seq-length 2048 \
     --grad-checkpoint \
     --seed 42

### Step 3.5: Testing the adapaters

In this step, I am doing the mathematical tests provided by the MLX community which include the **Loss** and the **Perplexity** tests.

The code for undertaking the testing of the model is as follows:

    ! mlx_lm.lora \
     --model ${mlx_path} \
     --data ${data} \
     --adapter-path ${adapters} \
     --test  

#### MLX Testing Metrics Analysis for Apple Silicon LLM Fine-tuning

##### Test Loss (Current Score: 3.945)
* **Definition**: Cross-entropy loss measured on the test set
* **Typical Range**: 1.5 to 5.0
* **Interpretation**:
  * < 2.5: Excellent performance
  * 2.5-3.5: Good performance
  * 3.5-4.5: Moderate performance
  * ">" 4.5: Poor performance
* **Current Value Assessment**: 3.945 indicates moderate performance

##### Test Perplexity (PPL) (Current Score: 51.684)
* **Definition**: Exponential of the loss (e^loss)
* **Purpose**: Measures model's uncertainty in predicting next tokens
* **Typical Range**: 10 to 100
* **Interpretation**:
  * < 20: Excellent performance
  * 20-40: Good performance
  * 40-60: Moderate performance
  * ">" 60: Poor performance
* **Current Value Assessment**: 51.684 indicates moderate uncertainty in predictions

#### Relationship Between Metrics
* Loss and perplexity are exponentially related (PPL = e^loss)
* Both metrics indicate prediction accuracy
* Lower values indicate better performance

#### Performance Assessment
The current results suggest moderate performance with potential for improvement through:
* Additional fine-tuning iterations
* Hyperparameter optimization
* Training data quality/quantity improvements
* Model architecture adjustments

#### Note
These metrics provide quantitative measures of model performance and can guide optimization efforts during the fine-tuning process.


## Part IV - Saving the fused model with the trained adapters & compression to GGUF format

## License

## Contribution
