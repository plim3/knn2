# Peter Lim G#00912242
# HW1 (CS484 - Spring 2019): Movie Review Sentiment Classification - KNN Classfier
# Run on Python 2.7.10

import nltk
import numpy as np
import time
from scipy.spatial import distance
import operator

start = time.clock()    # Start clock
localtime = time.asctime(time.localtime(time.time()))

k = 159 # K value (sqrt(n))
train_m = np.loadtxt(open("training_class.csv", "rb"), delimiter=",")	# Retrieve training set matrix
test_m = np.loadtxt(open("test_class.csv", "rb"), delimiter=",")	# Retrieve test set matrix
dic = {}	# A dictionary is used to associate the values from the matrix with their euclidean distances

with open('labels.dat', 'w') as f:
	for i in range(0,25000):
		q = [test_m[i][0],test_m[i][1],test_m[i][2],test_m[i][3]]	# positive, negative, neutral, compound scores from test set
		for j in range (0,25000):
			# Calculate euclidean distance
			p = [train_m[j][0],train_m[j][1],train_m[j][2],train_m[j][3]]	# positive, negative, neutral, compound scores from training set
			eu_dst = distance.euclidean(p, q)
			dic[j] = eu_dst
		sorted_d = sorted(dic.items(), key=operator.itemgetter(1)) # Sorts distances from smallest to largest

		# Find k nearest neighbors
		pos = 0
		neg = 0
		for a in range(0,k):
			if train_m[sorted_d[a][0]][4] == 1:	# Check the labels of the k nearest neighbors from the training set
				pos += 1
			else:
				neg += 1
		print "Review #", i+1
		print "Positive reviews: ", pos
		print "Negative reviews: ", neg

		# Write into a textfile -1/+1 labels
		if pos > neg:
			print "Labeling as positive review"
			f.write("+1\n")
		else:
			print "Labeling as negative review"
			f.write("-1\n")

		print "\n"
		#print "Sorted eu_dst: ", sorted_d
		#print dic


# End clock
print "Start time: ", localtime
endtime = time.clock()
localtime = time.asctime(time.localtime(time.time()))
print "End Time: ", localtime
print "\nTime Taken: ", (endtime - start)