import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
# library that helps handle datasets 
import pandas as pd 
# vader Sentiment Analysis model library 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

sid_obj = SentimentIntensityAnalyzer()

def sentiment_scores(sentence):

    sentiment_dict = sid_obj.polarity_scores(sentence)

    overall_sentiment = "neutral"
    
    if sentiment_dict['compound'] >= 0.05:
        overall_sentiment = "positive"
    elif sentiment_dict['compound'] <= -0.05:
        overall_sentiment = "negative"

    return {
        "Sentence": sentence,
        "Negative": sentiment_dict['neg'] * 100,
        "Neutral": sentiment_dict['neu'] * 100,
        "Positive": sentiment_dict['pos'] * 100,
        "Overall Sentiment": overall_sentiment,
        "Compound": sentiment_dict['compound']
    }



if __name__ == "__main__" :

    splits = {'train': 'train.jsonl', 'test': 'test.jsonl'}
    df = pd.read_json("hf://datasets/SetFit/tweet_sentiment_extraction/" + splits["test"], lines=True)
	
    df['Sentiment Results'] = df['text'].apply(sentiment_scores)

    sentiment_df = pd.DataFrame(df['Sentiment Results'].tolist())

    final_df = pd.concat([df, sentiment_df], axis=1).drop(columns=['Sentiment Results'])

    print(final_df)
    final_df.to_csv('sentiment_analysis_results.csv', index=False)
	
    y_true = label_binarize(final_df['label_text'], classes=['positive', 'neutral', 'negative'])
    
    # Use the VADER sentiment scores as the predicted probabilities for each class
    y_probs = final_df[['Positive', 'Neutral', 'Negative']].values / 100  # Normalizing to get probability scores

    # Calculate the AUC for the multi-class classification using the 'ovr' (one-vs-rest) strategy
    auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
    
    print(f'Macro-Averaged AUC: {auc:.4f}')

    sentiments = ['positive', 'neutral', 'negative']
    confusion_matrix = pd.DataFrame(np.zeros((3, 3), dtype=int), index=sentiments, columns=sentiments)

    # Populate the confusion matrix
    for _, row in final_df.iterrows():
        actual = row['label_text']
        predicted = row['Overall Sentiment']
        confusion_matrix.loc[actual, predicted] += 1

    
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=sentiments, yticklabels=sentiments)
    plt.title('VADER Model Sentiment Analysis Performance')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

