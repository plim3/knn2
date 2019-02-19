# Peter Lim G#00912242
# HW1 (CS484 - Spring 2019): Movie Review Sentiment Classification - Training Set
# Run on Python 2.7.10

import re
import nltk
import numpy as np
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# nltk.download('punkt')     # Used to tokenize words
# nltk.download('stopwords')      # Used to remove stop words

start = time.clock()    # Start clock
localtime = time.asctime(time.localtime(time.time()))

m = np.empty([25000,5])  # Initialize matrix
x = 0

# Read the training data file
with open('1548889051_0353532_train.dat', 'r') as f:
    for review in f:
        score = review[0]+review[1] # pos/neg review score
        m[x][4] = float(score)

        # Pre-processsing data
        review = re.sub('[^A-Za-z]', ' ', review)   # Removes none alphebetic characters
        review = review.lower()     # Change to lower case
        token_review = word_tokenize(review)    # Split words into a list
        print token_review.split()
        # bigrm = list(nltk.bigrams(token_review.split()))
        # print ', '.join(' '.join((a, b)) for a, b in bigrm)
        # print "Tokenized review before pp:\n", token_review
        y = 0
        for i in token_review:
            if len(i) == 1:   # Remove single characters
                token_review[y] = ""
            elif i == "br":   # Remove br's
                token_review[y] = ""
            elif i == "eof":  # Remove eof at the end
                token_review[y] = ""
            if i in stopwords.words('english'): # Remove stop words
                # print "Removing stopword: ", i
                token_review[y] = ""
            y += 1

        token_review = filter(None, token_review) # Filter the empty strings

        # print "Review after pp: \n", token_review

        join_review = ' '.join(word for word in token_review)   # Convert the list words into text
        # print "Joined review: ", join_review

        from nltk.sentiment.vader import SentimentIntensityAnalyzer     # Analyzes the text and provides scores of positivity and negativity
        sid = SentimentIntensityAnalyzer()
        result = sid.polarity_scores(review)
        print result
        m[x][0] = result['pos']     # Positive score
        m[x][1] = result['neg']     # Negative score
        m[x][2] = result['neu']     # Neutral score
        m[x][3] = result['compound'] # Compound score

        print m[x]
        print "Iteration: ", x + 1

        x += 1
        # if x == 100:
        break

    # from nltk.stem.porter import PorterStemmer  # Stemming
    # stemmer = PorterStemmer()

    # for i in range(len(token_review)):
    #     token_review[i] = stemmer.stem(token_review[i])

    # from autocorrect import spell   # Spell check
    # token_review[i] = stemmer.stem(spell(token_review[i]))

    # review_text = " ".join(token_review)

# Write predicted labels into text file
np.savetxt("training_class.csv", m, delimiter=",")   # Save as csv file

#End clock
print "Start time: ", localtime
endtime = time.clock()
localtime = time.asctime(time.localtime(time.time()))
print "End Time: ", localtime
print "\nTime Taken: ", (endtime - start)
