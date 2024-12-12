from datasets import load_dataset
import ollama
from tqdm.notebook import tqdm
import pandas as pd

def getResponse(inputString):
    response = ollama.chat(model='llama3.1', messages=[
        {
            'role': 'user',
            'content': inputString,
        }],
        options={'temperature':1.0 }
    )
    return response['message']['content']

ds = load_dataset("SetFit/tweet_sentiment_extraction")

answerConversion = {'P': 'positive', 'I': 'neutral', 'N': 'negative'}
acronymConversions = {'positive':'P', 'neutral':'I', 'negative':'N'}

def parseOutput(modelOutput):
    if (modelOutput[0] in ['P', 'I', 'N']) and (modelOutput[1:3] == ' -'):
        return answerConversion[modelOutput[0]]
    else:
        return -1
    


from collections import defaultdict
ds = load_dataset("SetFit/tweet_sentiment_extraction")
def get_first_n_per_label(dataset, n=3):
    label_dict = defaultdict(list)  
    

    for sample in dataset:
        label = sample['label']
        if len(label_dict[label]) < n:
            label_dict[label].append(sample)
        if all(len(v) >= n for v in label_dict.values()):
            break

    return label_dict


NUM_SAMPLES = 5
first_3_per_label = get_first_n_per_label(ds['train'], n=NUM_SAMPLES)

order = [2, 1, 0]  
interleaved_list = []
for i in range(NUM_SAMPLES):  
    for label in order:
        interleaved_list.append(first_3_per_label[label][i])

for sample in interleaved_list:
    print(sample['text'], sample['label_text'])





df_multiShot = pd.DataFrame()
df_multiShot.to_csv(f'test0_1_multishot_{NUM_SAMPLES}.csv')
print(len(interleaved_list))
featureEngineering = f'''[Instructions]: You are a sentiment analyzer tasked with classifying tweets. Given a question, compute the sentiment as P for Positive, I for neutral (indifferent) and N for negative. Give exactly one classification.'''

for i in range(len(interleaved_list)):
    featureEngineering = featureEngineering + f"\n[Example {i}] [Text] {interleaved_list[i]['text']} [Classification] {interleaved_list[i]['label_text']}"
featureEngineering = featureEngineering + '''\n[Instructions Repeated]: You are a sentiment analyzer tasked with classifying tweets. Given a question, compute the sentiment as P for Positive, I for neutral (indifferent) and N for negative. Give exactly one classification.
[Format]: Format your answer as: P OR I OR N - EXPLANATION HERE
[Question]: '''


print(featureEngineering)

# Initialize an empty list to store the outputs
outputs = []

# Loop through the samples with a progress bar
for i in tqdm(range(len(ds['test'])), desc="Processing Samples"):
    sample = ds['test'][i]
    modelOutput = getResponse(featureEngineering + sample['text'])
    parsedModelOutput = parseOutput(modelOutput)
    outputs.append([parsedModelOutput == sample['label_text'], parsedModelOutput, sample['label_text'], sample, modelOutput])
    
    # Append results to CSV every 150 iterations
    if (i + 1) % 50 == 0:
        # Convert outputs to a DataFrame
        outputsDF = pd.DataFrame(outputs, columns=['Correct', 'Predicted', 'Actual', 'Sample', 'Output'])
        
        # Append to the CSV file, don't overwrite existing data
        outputsDF.to_csv(f'test0_1_multishot_{NUM_SAMPLES}.csv', mode='a', header=not pd.read_csv(f'test0_1_multishot_{NUM_SAMPLES}.csv').shape[0], index=False)
        
        # Clear the outputs list after saving to free up memory
        outputs = []
    print(i)

# Save any remaining outputs that didn't get saved in the last chunk
if outputs:
    outputsDF = pd.DataFrame(outputs, columns=['Correct', 'Predicted', 'Actual', 'Sample', 'Output'])
    outputsDF.to_csv(f'test0_1_multishot_{NUM_SAMPLES}.csv', mode='a', header=not pd.read_csv(f'test0_1_multishot_{NUM_SAMPLES}.csv').shape[0], index=False)
