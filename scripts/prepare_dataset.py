from datasets import load_dataset

# Define the paths to your files

def create_hf_dataset():
    data_files = {}
    data_files["train"] = 'meQSum_Dataset\chq\\train.jsonl'
    data_files["test"]  = 'meQSum_Dataset\chq\\test.jsonl'
    data_files["validation"] = 'meQSum_Dataset\chq\\validate.jsonl'

    # Load the datasets
    dataset = load_dataset('json', data_files=data_files)

    # Print the first example of the training dataset
    print(dataset)
    
    return dataset

# check the number of examples in each dataset