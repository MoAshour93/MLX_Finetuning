# 🚀 Fine-tuning Large Language Models on Apple Silicon Macs using MLX & Llama.cpp

This repository aims at providing detailed guidance about how you can fine-tune large language models on Apple Silicon Macs using the **MLX Framework** and **llama.cpp**.

## 👨‍💻 Developer Contact
- **Name:** Mohamed Ashour
- **Email:** mo_ashour1@outlook.com
- **LinkedIn:** [Mohamed Ashour](https://www.linkedin.com/in/mohamed-ashour-0727/)

## 🙏 Acknowledgements
A number of resources helped in creating this guidance. This repository aims at collating the most useful information from all the following resources:
* [The MLX community](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md)
* Andy Peating articles:
    * [Part 1 - Setting up your environment](https://apeatling.com/articles/part-1-setting-up-your-environment/)
    * [Part 2 - Building your training data for fine-tuning](https://apeatling.com/articles/part-2-building-your-training-data-for-fine-tuning/)
    * [Part 3 - Fine-tuning your llm using the mlx framework](https://apeatling.com/articles/part-3-fine-tuning-your-llm-using-the-mlx-framework/)
    * [Part 4- Testing and interacting with your fine-tuned LLM](https://apeatling.com/articles/part-4-testing-and-interacting-with-your-fine-tuned-llm/)
* [Llama 3 Model cards and prompting format](https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/)
* [Fine-tuning LLMs on Mac OS using MLX and run with Ollama](https://medium.com/rahasak/fine-tuning-llms-on-macos-using-mlx-and-run-with-ollama-182a20f1fd2c)

## 📁 Repository Main Files
* **Readme.md** - This file explains the repository, components and way to undertake fine-tuning
* **Convert_CSV_to_JSONL.py** - Converts csv training dataset into train, test & valid files
* **Mistral_Instruct_MLX_Fine-tuning.ipynb** - Main jupyter notebook for fine-tuning, testing, and conversion
* **ModelFile** - Contains chat template and system prompt for Ollama usage

## 📑 Table of Contents
* [Part I - Setting the Coding Environment](#part-i---setting-the-coding-environment)
    * [Step 1.1 - Opening Terminal](#step-11)
    * [Step 1.2 - Installing Homebrew](#step-12)
    * [Step 1.3 - Setting PATH](#step-13)
    * [Step 1.4 - Installing Git](#step-14)
    * [Step 1.5 - Cloning MLX Repository](#step-15)
    * [Step 1.6 - Changing Directory](#step-16)
    * [Step 1.7 - Installing Python](#step-17)
    * [Step 1.8 - Installing Pip](#step-18)
    * [Step 1.9 - Installing Requirements](#step-19)
    * [Step 1.10 - Setting up Llama.cpp](#step-110)

* [Part II - Create/Import Data for Finetuning](#part-ii---createimport-data-for-finetuning)
    * [Step 2.1 - Preparing CSV Data](#step-21)
    * [Step 2.2 - Understanding Chat Templates](#step-22)
    * [Step 2.3 - Converting to JSONL Format](#step-23)

* [Part III - Training the Model, Testing and Validation](#part-iii---training-the-model-testing-and-validation)
    * [Step 3.1 - Defining Variables](#step-31)
    * [Step 3.2 - Downloading Model](#step-32)
    * [Step 3.3 - Converting to MLX Format](#step-33)
    * [Step 3.4 - Fine-tuning Process](#step-34)
    * [Step 3.5 - Testing Adapters](#step-35)
    * [Step 3.6 - Generating Responses](#step-36)

* [Part IV - Saving and Converting the Model](#part-iv---saving-the-fused-model-with-the-trained-adapters--compression-to-gguf-format)
    * [Step 4.1 - Fusing Model](#step-41)
    * [Step 4.2 - Testing Fused Model](#step-42)
    * [Step 4.3 - MLX Inferencing](#step-43)
    * [Step 4.4 - Huggingface Export](#step-44)
    * [Step 4.5 - GGUF Conversion](#step-45)
    * [Step 4.6 - Ollama Export](#step-46)

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

In this step, I am trying to make the best use of the Low Rank Adaptation with mlx_lm.lora under the mlx_lm package to fine-tune the already converted MLX Mistral V0.3. The summary of the fine-tuning code is as shown below. 

 

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
Please refer to the jupyter notebook which includes the full explanation of the arguments of the fine-tuning.

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

### Step 3.6: Generating responses from the fine-tuned adapters

Having tested the model above, it is now time to generate some responses to see how good the trained adapters really are. The MLX framework allows for inferencing using the base model and the trained adapeters as shown in the code below.

    !mlx_lm.generate \
       --model ${mlx_path} \
       --adapter-path ${adapters} \
       --system-prompt "${System_prompt}" \
       --prompt "Give me a good quality quality example of competency Procurement and Tendering level 2 " \
       --max-tokens 400 \
       --temp  0.3 \
       --use-default-chat-template \
       --verbose True

**The response obtained from the inferencing is as shown below:**
* During my employment at my current employer/one of the companies that I worked in, I have been involved in a number of Procurement and Tendering routes including Single-Stage Tender, Two-Stage Tender, Negotiated Tender, Framework Agreement, and Selective Tendering.
* One of my recent projects was tendered using a Two-Stage Tender process. The first stage of the tender process required potential Contractors to submit a tender package containing a completed tender price statement, design and construction information, and health and safety information. Following the assessment of the tender submissions by a tender evaluation panel, four pre-qualified Contractors were invited to submit a second stage tender package containing a price change reflecting any amendments made to the tender documents, updated construction programme, and a tender cost summary in line with the tender brief. Following the assessment of the second stage tender submissions, a contract was awarded with one of the four pre-qualified Contractors.
* I have also been involved in the preparation of briefing documents for the Procurement team on a number of projects. These documents included Tender Briefing and Tender Document Checklists.
* I have also gained an understanding of the different Procurement routes including Traditional Procurement, Design and Build, Construction Management, and Management Contracting.

**The reponse metrics are as shown below:**
* Prompt: 269 tokens, 631.665 tokens-per-sec
* Generation: 282 tokens, 22.400 tokens-per-sec
* Peak memory: 14.678 GB

***The Jupyter notebook contains a couple of exmaples obtained from the inferencing to showcase how good the model really is.***

## Part IV - Saving the fused model with the trained adapters & compression to GGUF format

In this part of the code we are going to explore how we can make the best use of the fuse functionlity under the MLX Library

### Step 4.1 - Fusing the MLX model with the trained adapters and saving it locally

In this step, I am trying to fuse the trained adapters with the MLX converted model after having seen the performance.

#### Purpose of the Command
This command performs three main operations:
1. Loads the original base model
2. Incorporates the fine-tuned LoRA adaptations
3. Creates a new, standalone model with merged weights in full precision

#### Common Use Cases
* Creating deployment-ready models
* Preparing models for different platforms
* Converting fine-tuned models to full precision
* Generating models for scenarios requiring maximum accuracy

#### Note
The resulting model will:
* Be larger in size due to de-quantization
* Include all fine-tuned adaptations
* Be ready for direct use without needing separate adapter loading

#### Code
Fusing the LoRA adapters with the MLX converted model could be undertaken using the following code:

    !mlx_lm.fuse \
        --model {mlx_path} \
        --save-path {output_directory} \
        --adapter-path {adapters} \
        --de-quantize

The Jupyter notebook contains further explanation for every argument in the code above.

### Step 4.2: Testing the fused model for correctness

In the code block below, I am testing the fused MLX model with the trained adapters to make sure that the model after the fuse process is behaving as it should.

For this I am using the resultant MLX-finetuned model with the adaptes from **Step 4.1** above.

For further details about the **mlx_lm.generate** function, please refer to **Step 3.6** above.

**The code related to this step is as shown below:**

    !mlx_lm.generate \
        --model {output_directory} \
        --system-prompt "{System_prompt}" \
        --prompt "Give me a good quality quality example of competency: Construction Technology level 3 " \
        --max-tokens 400 \
        --temp  0.1 \
        --use-default-chat-template \
        --verbose True

**The response received is as follows:**

* I have been involved in numerous projects where I had to advise the client on various matters relating to the construction of the building. For example, when reviewing the cost report prepared by the loss leader consultants on a commercial scheme, I was able to advise the client about the correctness of the figures being claimed by the claimant contractor and what construction methods were in accordance with local building regulations as opposed to the claimant entitlement to additional costs.
I was able to further advise the client about the works which were within spec and which works were in accordance with the contract (both contract terms which were fair and reasonable, as well terms which were in line with current market terms). I have also been able to advise clients about the correct contractual wording which would class the works as both a change to the contract and/or change to the spec.
* I have been able to advise clients about the correct contractual wording required to establish whether the changes were within scope of the contract or not. I have also been able to advise clients about the correct contractual wording required to establish whether the changes were within scope of the contract and who the correct party to be claimed against.
In terms of contractual claims pertaining to liquidated damages, I have been able to advise clients about the correct calculation of liquidated damages and whether the claimant was entitled to claim additional damages after the liquidated period had ended.
* In terms of extensions to the contract, I have been able to advise clients about the correct calculation of extensions and whether the contract allowed for the claimed extensions.
* In terms of extra work orders, I have been able to advise clients about the correct calculation of claimed variations and whether the claimed rates were in line with current market rates.
* In terms of interim payments, I have been able to advise clients about the correct calculation of retention release and whether the contract allowed for the claimed retention period.
In terms of project set-up,


**The reponse metrics are as shown below:**
* Prompt: 264 tokens, 643.068 tokens-per-sec
* Generation: 400 tokens, 25.764 tokens-per-sec
* Peak memory: 14.654 GB

### Step 4.3: Inferencing the fused fine-tuned model using the MLX community guidelines

This step is optional. I included it within my code to showcase how you can use the fused MLX Fine-tuned model with its trained adapters inside a python environment using the MLX community Huggingface guidelines.

You can refer to the location of the fine-tuned model which contains the safetensors for the model weights, the tokenizers, the configuration and other relevant files that are necessary for inferencing.

You can tweak the prompt however you see fit. It could be a list of prompts that need to be dealt with in one hit.

The MLX community guidelines for inferencing using the model with safetensor files are as follows:

     from mlx_lm import load, generate
     
     fine_tuned_mlx_model = output_directory.strip("'")
     model, tokenizer = load(fine_tuned_mlx_model)
     prompt="Give me a good quality quality example of competency: Construction Technology level 3"
     
     if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
       messages = [{"role": "user", "content": prompt}]
       prompt = tokenizer.apply_chat_template(
       messages, tokenize=False, add_generation_prompt=True)

     response = generate(model, tokenizer, prompt=prompt, verbose=True)

### Step 4.4: Exporting the model to Huggingface for further conversion to GGUF

This step is optional. It tackles the process of saving LLMs on the Huggingface website which can then be transformed into a GGUF format using the **GGUF my repo** initiative on the Huggingface website.

Once you have uploaded your fine-tuned model to your repository on the Huggingface website, you could go to <https://huggingface.co/spaces/ggml-org/gguf-my-repo> and then you could refer to your repository and choose what quantization level suits you the most.

***You can do that export using the following block of code:***
     
    !mlx_lm.fuse \
        --model ${mlx_path} \
        --adapter-path ${adapters} \
        --hf-path ${hf_model} \
        --upload-repo ${hf_upload_repo} \
        --de-quantize

### Step 4.5: Converting the fused finetuned model with the Low Rank Adaptors using llama.cpp

In this step we are going to use the functionality of converting huggingface models to GGUF under the **llama.cpp**.

To do that we have to open the terminal where **llama.cpp** is saved. We can do that by going to the folder where **llama.cpp** is cloned and then right click on the pathname bar and choose the option ***Open in Terminal***.

This will open a terminal instance in the folder llama.cpp is cloned. We can then make the best use of the **convert_hf_to_gguf** python file available in this directory for our conversion process.

For the purpose of this notebook, I have saved the location where llama.cpp is cloned on my machine in a variable called "llama_cpp_path" which you could find in **Part III , Step 3.1** above.  

This conversion to GGUF is important in case we are going to use the models in third party applications such as ***Ollama*** and ***Open WebUI***.

**Following the explanation above, I have undertaken the following:**
1. Referring to the location where I saved my fine-tuned model
2. Referring to the location of where I want to save the resultant GGUF model 
3. Stating explicitly what level of quantization is required. You can refer to the explanation of the code arguments in the previous code block.
4. Using the no lazy which forces immediate loading of all model weights instead of loading them when needed, ensuring complete model validation upfront
5. I am also enabling the verbose option which enables detailed output logging during conversion, showing step-by-step progress and additional information

***Here is the code for what is explained above***

    %cd {llama_cpp_path}
    !python3 convert_hf_to_gguf.py \
        {output_directory} \
        --outfile {output_directory} \
        --outtype q8_0 \
        --no-lazy \
        --verbose 

### Step 4.6: Exporting the Fine-tuned model with GGUF to Ollama for further usage using the Ollama API

In this step, I am creating a model file for the fine-tuned MLX Mistral model, taking into account its architecture.

Every model has different architecture. One has to respect the prompting architecture in order to get meaningful inferencing.

I created the model file taking into account how the parameters, the template and the system prompt should be tweaked for the Mistral Architecture.

In order to create a model file, you can create an empty txt file and then make sure to remove its extension. I have named mine "ModelFile"

There are 4 main items that you need to account for when creating a ModelFile for a LLM to be used under **Ollama**:

**1 - From:** 
* You have to state the directory where your gguf file is saved so that the model file can relate to it.

**2 - Parameters:**
* Here you can type in the parameters that you want to include within your model architecure. 
* I have also included the start and stop tokens that can help the model produce meaningful responses and not to produce text indefinitely.

**3 - Template:**
* The template allows for meaningful chats with the model. You can see how the chat is display by displaying the model template card on Ollama's website.
* The template that is used below is for the Mistral Architecture.    

**4- System Prompt:**
* I have chosen to include a system prompt to make the responses provided by the model quite relevant to its use-case and also aiming to reduce hallucination.
* System Prompts can also allow more control over the model in the way it behaves, where it gets its information from, what are the boundaries,..etc

***You can refer to the uploaded ModelFile within the repository for further details. The jupyter notebook also contains the full explanation for the ModelFile components.***

Once we have created the Model File for our desired model, it is now time to export our MLX Fine-tuned model into **Ollama**.

* In order to export a model to ollama, you have to open the terminal where the gguf file is stored.
* We can do that by going to the folder where the gguf file is stored and then right click on the comprising folder on the pathname bar and choose the option ***Open in Terminal***.
* For the purpose of this notebook, I have kept it simple. I stored the directory where the gguf file is saved in a variable called ***output_directory***. You can check the full list of variables in ***Part III, Step 3.6*** above.
* The export option could be done using the code : 
      
      %cd {output_directory} 
      !ollama create {Your_Desired_Model_Name} -f ModelFile
## 📜 License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

### Apache License 2.0 Summary:
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ℹ️ License and copyright notice required
- ℹ️ State changes must be documented
- ℹ️ Changes made must be open source

## 🤝 Contribution
Contributions are welcome! Please feel free to submit a Pull Request.
