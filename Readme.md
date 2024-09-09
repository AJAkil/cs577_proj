The repository contains the datasets and codes for cs577 Project titled **Prompt-Enhanced Medical Question Summarization**

The folders are organized in the following way:
- **meQSum_Dataset**: Consists of the original meQSum Dataset splitted into train, validation and test sets
- **ner_tagged_dataset**: Consists of the NER augmented meQSum dataset
- **coocccur_dataset**: Consists of cooccurance tag augmented meQSum dataset
- **notebooks**: Consist of finetuning, zeroshot generation and zero_shot_co-occurence notebooks. 
- **scripts**:  Consists of scripts to create the NER tagged dataset, generating all unqiue NER tags and preparaing the datasets for huggingface library.

All finetuning of model flant5-base has been done with huggingface and logs been generated with wandb. The training was done in google colab pro with L4 GPU with 22.5GB of VRam in High RAM environment. Required modules to run the notebooks have been mentioned in the notebooks. To run, training or zero shot inference, simply use the notebooks in the **notebooks** folder.

You can find the project report pdf with details of our experiments with different prompt strategy and finetuning in the ***prompt_enhanced_medical_question_summarization_report.pdf***
