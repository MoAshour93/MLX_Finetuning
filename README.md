# MLX_Finetuning
This repository aims at providing a detailed guidance about how you can fine-tune large language models on Apple Silicon Macs using MLX and llamac.cpp.

#Acknowledgements
A number of resources helped in creating this guidance. This repository aims at collating the most useful information from all the following resources:
* The MLX community - https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md
* Andy Peating articles:
    * Part 1 - Setting up your environment- https://apeatling.com/articles/part-1-setting-up-your-environment/
    * Part 2 - Building your training data for fine-tuning  https://apeatling.com/articles/part-2-building-your-training-data-for-fine-tuning/
    * Part 3 - Fine-tuning your llm using the mlx framework https://apeatling.com/articles/part-3-fine-tuning-your-llm-using-the-mlx-framework/
    * Part 4- Testing and interacting with your fine-tuned LLM  https://apeatling.com/articles/part-4-testing-and-interacting-with-your-fine-tuned-llm/
* Llama 3 Model cards and prompting format available through : https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/
* Fine-tuning LLMs on Mac OS using MLX and run with Ollama - https://medium.com/rahasak/fine-tuning-llms-on-macos-using-mlx-and-run-with-ollama-182a20f1fd2c

Please note that different models have different model cards and prompting templates and you should visit the developer of every LLM to tweak the code to account for the relevant chat template.
