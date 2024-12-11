#!/usr/bin/env python
# coding: utf-8

# # MT5 Model Training and Testing

# In[4]:


# !python.exe -m pip install --upgrade pip --user
# !pip install transformers --user
# !pip install datasets --user
# !pip install torch --user
# !pip install scikit-learn --user
# !pip install sentencepiece --user
# !pip install transformers[torch] --user


# # Setting up the Data

# In[2]:


import pandas as pd
from transformers import Trainer, TrainingArguments, MT5Tokenizer, MT5ForConditionalGeneration
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch


# In[3]:


train_df = pd.read_csv('C:\PycharmProjects\CS 534\mt5Model\\tweet-sentiment-extraction\\train.csv')
test_df = pd.read_csv('C:\PycharmProjects\CS 534\mt5Model\\tweet-sentiment-extraction\\test.csv')


# In[4]:


from sklearn.model_selection import train_test_split

# Define the input and output
train_df['input_text'] = train_df['text']
train_df['output_text'] = train_df['sentiment']

# Split the training data into sections for training the model and sections for validating the model
# The current setup has 90% of the data being used as training data and 10% of the data being used to validate the model
# X_train is the training data input values and x_val is the validation data input values
# y_train is the training data output values and y_val is the validation data output values
X_train, X_val, y_train, y_val = train_test_split(
    train_df['input_text'],
    train_df['output_text'],
    test_size=0.1,
    random_state=42
)


# In[5]:




tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')

def encode_data(texts):
    if isinstance(texts, pd.Series):  # If input is a Pandas Series
        texts = texts.tolist()  # Convert to list
    elif isinstance(texts, str):  # If input is a single string
        texts = [texts]  # Make it a list
    else:
        # If it gets here, then there is a massive error, but just want to make sure
        print("\n\n\nHuuuuuge Error Here\n\n\n")
        print(type(texts))

    # Ensure all elements are strings and handle missing values
    # set text to a string, and if the string has values then do nothing, otherwise, set the string spots 
    #to "NA" for every spot in the string
    for i in range(len(texts)):
        if pd.notna(texts[i]):
            texts[i] = str(texts[i])
        else:
            texts[i] = "NA"
    # texts = [str(text) if pd.notna(text) else "NA" for text in texts]

    # This code converts the whole text thing into tensors (torch lists of same size) which are 
    #number the number conversions of each word in a string
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=50,
        return_tensors='pt'
    )

#tokenize the training data and the validating data
train_encodings = encode_data(X_train)
val_encodings = encode_data(X_val)

print(type(X_train))  # Should be a Pandas Series
print(X_train)  # Should show some strings

print("\n\n")

print(y_train)

#test_encodings = encode_data(test_df['text'])


# In[6]:




# class for processing tweet datasets and turning it into usable data (Tokenize)
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.encodings.items()}

        # Tokenize labels separately and ensure they are padded
        label_encoding = tokenizer(
            self.labels[index],
            padding='max_length',  # Pad to max length
            truncation=True,
            max_length=50,  # You can set this to the maximum length you expect
            return_tensors='pt'
        )

        item['labels'] = label_encoding['input_ids'].squeeze()  # Squeeze to remove unnecessary dimensions
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TweetDataset(train_encodings, y_train.values)
val_dataset = TweetDataset(val_encodings, y_val.values)


# print("")
# print(train_dataset[390].get(0))
# print("")
# print(train_dataset[390].get(1))

# for idx in range(len(train_dataset)):
#     item = train_dataset[idx]  # Get item at index idx
#     input_ids = item['input_ids']
#     attention_mask = item['attention_mask']
#     label = item['labels']

#     print(f"Input IDs: {input_ids}")
#     print(f"Attention Mask: {attention_mask}")
#     print(f"Label: {label}")

# print("")        #check some of the values manually to make sure they are all right
# print(train_dataset[0])  # Check the first item
# print(train_dataset[1])  # Check the second item
# print("\n\n\n\nThe 390s start here\n\n\n\n")
# print(train_dataset[390])
# print("\n\n")
# print(train_dataset[391]) 
# print("\n\n")
# print(train_dataset[392]) 
# print("\n\n")
# print(train_dataset[393])
# print("\n\n")
# print(train_dataset[394]) 
# print("\n\n")
# print(train_dataset[395])
# print("\n\n")
# print(train_dataset[396]) 
# print("\n\n")
# print(train_dataset[397])
# print("\n\n")
# print(train_dataset[398]) 
# print("\n\n")
# print(train_dataset[399])
# print("\n\n")
# print(train_dataset[400])


# In[8]:





model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)


# Trains the data and then tests the trained model to get the best results
# train_dataset is the list/tensor data which contains 90% of the training data to train the model
# val_dataset is the list/tensor data whic contains th remaining 10% of the training data to train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# import time

# print("here")

# counter = 0
# for text, string in zip(train_dataset, texts):
#     if(counter == 9):
#         counter = 0
#         time.sleep(10)
#     print(text)
#     print(string)
#     counter += 1

trainer.train()


# # Save Model and Tokenizer

# In[10]:


trainer.save_model('./justTrainedModel')
tokenizer.save_pretrained('./JustTrainedTokenizer')


# # Generate Predictions for Test Set and Examine Accuracy

# In[12]:


# Compare predictions with true labels
true_labels = test_df['sentiment'].tolist()
print(true_labels)


# In[34]:


#maybe this version will work better

test_encodings = encode_data(test_df['text'])
test_dataset = TweetDataset(test_encodings, test_df['sentiment'])  # Tokenizing data


# In[11]:


# Make predictions
predictions = trainer.predict(test_dataset)

# Decode the predictions
decoded_predictions = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)


# In[ ]:


# Print the inputs and predictions
for text, predicted in zip(test_df['text'], decoded_predictions):
    print(f"Input: {text}\nPredicted: {predicted}\n")


# In[ ]:


#see how right/wrong it was
for text, actual, predicted in zip(test_df['text'], test_df['sentiment'], predictions):
    print(f"Input: {text}\nExpected: {actual}\nPredicted: {predicted}\n")


# In[ ]:


#Convert the predictions to actual langauge
predicted_texts = tokenizer.batch_decode(predictedSentiment.predictions, skip_special_tokens=True)

# Print the predictions
for text, pred in zip(test_texts, predictions):
    print(f"Input: {text}\nPrediction: {pred}\n")


# # Code to Run Before Tweet Analysis

# In[15]:


from transformers import MT5Tokenizer, MT5ForConditionalGeneration, Trainer, TrainingArguments
import torch
import pandas as pd
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


# # Load Previously Pretrained Models (If Model was Unloaded or Before Election Analysis)

# In[18]:


# Load the model
trained_model = MT5ForConditionalGeneration.from_pretrained('./justTrainedModel')

# Load the tokenizer
trained_tokenizer = MT5Tokenizer.from_pretrained('./JustTrainedTokenizer')


# # Running Model on Trump Election Tweets

# In[19]:


election_tweets_df = pd.read_csv('C:\\Users\\rhunt\\Downloads\\trump_harris_tweets.csv')


# In[20]:


trump_results = list()

for cell in election_tweets_df['trump']:
    sample_tweet = cell
    # Example: Let's generate text for a few sample tweets
    # sample_tweet = election_tweets_df['trump'][0]

    # Tokenize the input text
    input_ids = trained_tokenizer.encode(sample_tweet, return_tensors="pt")

    # Generate sentiments (in an encoded form)
    generated_ids = trained_model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

    # Decode the sentiments
    generated_text = trained_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # add sentiment results to list of sentiments
    trump_results.append(generated_text)

    print("Generated Text: ", generated_text)
print("\n All...\n ", trump_results)


# # Print Trump Tweet Results 

# In[49]:


# Print the inputs and predictions
for text, predicted in zip(election_tweets_df['trump'], trump_results):
    print(f"Input: {text}\nPredicted: {predicted}\n")


# # Running Model on Harris Election Tweets

# In[50]:


harris_results = list()

for cell in election_tweets_df['harris']:
    sample_tweet = cell
    # Example: Let's generate text for a few sample tweets
    # sample_tweet = election_tweets_df['trump'][0]

    # Step 1: Tokenize the input text
    input_ids = trained_tokenizer.encode(sample_tweet, return_tensors="pt")

    # Step 2: Use the model's generate function to generate text
    # You can set parameters such as max_length, num_beams, do_sample, etc., for controlling the output
    generated_ids = trained_model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

    # Step 3: Decode the generated tokens back into text
    generated_text = trained_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    harris_results.append(generated_text)

    print("Generated Text: ", generated_text)
print("\n All...\n ", harris_results)


# # Print Harris Tweet Results

# In[51]:


# Print the inputs and predictions
for text, predicted in zip(election_tweets_df['harris'], harris_results):
    print(f"Input: {text}\nPredicted: {predicted}\n")


