# Peter Lim G#00912242
# HW1 (CS484 - Spring 2019): Movie Review Sentiment Classification - Test Set
# Run on Python 2.7.10

import re
import nltk
import numpy as np
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

start = time.clock()    # Start clock
localtime = time.asctime(time.localtime(time.time()))

m = np.empty([25000,5])  # Initialize matrix
x = 0

# Read test data file for prediction
with open('1548889052_1314285_test.dat', 'r') as f:
    for review in f:

    	# print "Review: \n", review
        # Pre-processsing data
        review = re.sub('[^A-Za-z]', ' ', review)   # Removes none alphebetic characters
        review = review.lower()     # Change to lower case
        token_review = word_tokenize(review)    # Split words into a list
        # print "Tokenized review before pp:\n", token_review
        y = 0
        for i in token_review:
            if len(i) == 1:   # Remove single characters
            	# print "Removing single char: ", i
                token_review[y] = ""
            elif i == "br":   # Remove br's
            	# print "Removing br: ", i
                token_review[y] = ""
            elif i == "eof":  # Remove eof at the end
            	# print "Removing eof: ", i
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

np.savetxt("test_class.csv", m, delimiter=",")   # Save as csv file

# End clock
print "Start time: ", localtime
endtime = time.clock()
localtime = time.asctime(time.localtime(time.time()))
print "End Time: ", localtime
print "\nTime Taken: ", (endtime - start)