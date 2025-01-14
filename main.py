import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords

def ngram_getter():
    stop = stopwords.words('english')
    HSD_scores = pd.read_csv("All_Teams_Labels_Updated.csv")
    HSD_scores['Sentence'] = HSD_scores['Sentence'].str.lower()
    for word in stop:
        exact_match = word + ' '
        HSD_scores['Sentence'] = HSD_scores['Sentence'].str.replace(exact_match, ' ')
    texts = HSD_scores['Sentence']
    print(texts)
    labels = [0, 1]
    vectorizer = CountVectorizer(ngram_range=(1, 3))
    vectorizer.fit(texts)  # build ngram dictionary

    # vectorize texts into bag of words
    ngrams = vectorizer.transform(texts)

    print(ngrams.todense())
    keys_values_sorted = sorted(list(vectorizer.vocabulary_.items()), key=lambda t: t[1])
    keys_sorted = list(zip(*keys_values_sorted))[0]
    ngrams_matrix = ngrams.todense()
    df = pd.DataFrame(ngrams_matrix, columns=keys_sorted)
    max_values = df.max()

    # Step 3: Filter the columns where the maximum value is greater than 1
    # columns_with_max_gt_1 = max_values[max_values > 10].index
    #
    # # Step 4: Select those columns from the DataFrame
    # filtered_df = df[columns_with_max_gt_1]
    filtered_df = df.stack()[df.stack() > 5]

    # Convert the Series to a DataFrame
    filtered_df = filtered_df.reset_index()
    filtered_df.columns = ['Row', 'Column', 'Value']

    # Save the DataFrame to an Excel file
    file_path = 'ngrams.xlsx'
    filtered_df.to_excel(file_path, index=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ngram_getter()
