import pandas as pd
import requests
import seaborn as sns
import json
from string import punctuation
import nltk  # Need to pip to install the package first
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords  # Then, import them to use
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def classificationModelPerformanceEvaluator(X, y, classifier, numOfRun, testSize):
    performanceList = []
    n = 1
    while n <= numOfRun:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=None)
        classifier.fit(X_train, y_train)
        prediction = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, prediction)
        performanceList.append(accuracy)
        n = n + 1
    return performanceList


# Business Search      'https://api.yelp.com/v3/businesses/search'
# Business Reviews     'https://api.yelp.com/v3/businesses/{id}/reviews'

if __name__ == '__main__':
    API_KEY = 'RMuHGUQJKLRVz3uug59WjJ-poq9x_LVA5Q0dF2qjNhMLwjGjqODaaEH5rstxoFg0BeZWhtRwYWcFnHT425Vc3kuc6cPkEPGGUmxJ6QMEUFDMbWNs_r3zpocBVeCgY3Yx'
    ENDPOINT = 'https://api.yelp.com/v3/businesses/search'
    HEADERS = {'Authorization': 'bearer %s' % API_KEY}
    businessIDList = []
    offset = 0
    n = 1
    # Get 900 Business information, since later each business will spend 1 API call to retrieve review data,
    # and we cannot exceed 1000 calls in total
    while offset <= 850:
        print("Getting information for page " + str(n))
        PARAMETERS = {'term': 'restaurant',
                      'limit': 50,
                      'offset': offset,
                      'radius': 10000,
                      'location': 'Toronto'}
        response = requests.get(url=ENDPOINT, params=PARAMETERS, headers=HEADERS)
        responseDic = response.json()
        for business in responseDic["businesses"]:
            businessIDList.append(business['id'])
        offset = offset + 50
        n = n + 1
    # Now we retrieve the review of these businesses
    reviewList = []
    ratingList = []
    n = 1
    for business in businessIDList:
        print("Getting information for business " + str(n))
        businessAddress = "https://api.yelp.com/v3/businesses/" + business + "/reviews"
        ENDPOINT = businessAddress
        PARAMETERS = {'locale': 'en_CA',
                      'limit': 3}
        response = requests.get(url=ENDPOINT, params=PARAMETERS, headers=HEADERS)
        responseDic = response.json()
        for review in responseDic["reviews"]:
            reviewList.append(review['text'])
            ratingList.append(review['rating'])
        n = n + 1
    # Save the reviews and ratings in to a data table
    review_table = pd.DataFrame({"review": reviewList,
                                 "rating": ratingList})
    # Check distribution of the ratings
    sns.countplot(x=review_table["rating"])
    plt.title("Rating Distribution")
    plt.show()
    # Export the datatable to csv
    review_table.to_csv("toronto_business_review.csv", index=False)

    # Reload the review table using the csv, so later we can run from here
    review_table = pd.read_csv("toronto_business_review.csv")
    # Change rating to positive/negative and drop ratings column
    sentiment = []
    for rating in review_table["rating"]:
        if rating >= 4:
            sentiment.append("positive")
        else:
            sentiment.append("negative")
    review_table["sentiment"] = sentiment
    review_table.drop("rating", axis="columns", inplace=True)
    reviewTableList = review_table.values.tolist()
    # Save result back to JSON object
    reviewJSON = json.dumps(reviewTableList)
    # Create a JSON file
    jsonFile = open("review.json", "w")
    jsonFile.write(reviewJSON)
    jsonFile.close()
    # Text mining model
    nltk.download("stopwords")  # Download all the stopwords first
    englishStopWords = set(stopwords.words('english'))


    def cleanText(text):
        tokens = text.split()
        transTable = str.maketrans('', '', punctuation)
        tokenTemp = []
        for word in tokens:
            tokenTemp.append(word.translate(transTable))
        tokens = tokenTemp
        tokens = [word for word in tokens if word.isalpha()]
        # Now we lower case all capital letters
        tokens = [word.lower() for word in tokens]
        # Remove stop words
        tokens = [word for word in tokens if word not in englishStopWords]
        # Get rid of words that are too short
        tokens = [word for word in tokens if len(word) > 1]
        return ' '.join(tokens)


    listOfSentence = []
    for sentence in review_table["review"]:
        listOfSentence.append(cleanText(sentence))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(listOfSentence)
    tokenizer.word_index  # tokenizer.word_index returns a dictionary object
    wordList = list(tokenizer.word_index)  # tokenizer.word_index returns a dictionary object
    wordBagCount = tokenizer.texts_to_matrix(listOfSentence, mode='count')
    resultDataframeCount = pd.DataFrame(wordBagCount)
    resultDataframeCount.columns = ["Dummy"] + wordList
    resultDataframeCount.drop("Dummy", axis="columns", inplace=True)
    X = resultDataframeCount
    y = review_table["sentiment"]

    # Train a Multinomial Naive Bayes prediction model
    mnb = MultinomialNB()
    performanceList = classificationModelPerformanceEvaluator(X=X, y=y, classifier=mnb, numOfRun=30, testSize=0.2)
    sns.boxplot(data=performanceList)
    plt.title('Model Performance', fontsize=20)
    plt.show()
