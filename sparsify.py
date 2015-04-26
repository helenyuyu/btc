from scipy import sparse
import numpy as np
from sklearn.decomposition import TruncatedSVD, RandomizedPCA, NMF
from sklearn.metrics import matthews_corrcoef, roc_auc_score, roc_curve, auc
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from functools import partial
import math
from lda import LDA

num_users = 444075
f = open('txTripletsCounts.txt', 'r')
triplets = f.readlines()
triplets = np.array([l.split() for l in triplets])
row = [int(s) for s in triplets[:,0]]
col = [int(s) for s in triplets[:,1]]
data = [int(s) for s in triplets[:,2]]
binary = False
if binary:
  data = [1 for s in triplets[:,2]]
X = sparse.csc_matrix((data, (row,col)), shape=(num_users, num_users))
f.close()


def nmf():
  name = "nmf"
  model = NMF(n_components = 25)
  if binary:
    print "binary", name
  else:
    print "counts", name
  model.fit(X)
  A_T = sparse.csc_matrix(model.components_)
  Z = A_T*X
  A = A_T.transpose()
  def get_prob(i, j):
    return (A.getrow(i)*Z.getcol(j))[0,0]
  display_results(name, get_prob) 


def svd():
  name = "svd"
  if binary:
    print "binary", name
  else:
    print "counts", name
  model = TruncatedSVD(n_components = 25)
  model.fit(X)
  print(model.explained_variance_ratio_)
  print "total explained variance", model.explained_variance_ratio_.sum()
  A_T = sparse.csc_matrix(model.components_)
  Z = A_T*X
  A = A_T.transpose()
  def get_prob(i, j):
    return (A.getrow(i)*Z.getcol(j))[0,0]
  display_results(name, get_prob)



def kmeans():
  name = "kmeans"
  if binary:
    print "binary", name
  else:
    print "counts", name
  model = KMeans()
  labels = model.fit_predict(X)
  def get_prob(i,j):
    return model.cluster_centers_[labels[i]][j]
  display_results(name, get_prob)

def lda():
  name = "lda_py"
  n_topics = 20
  model = LDA(n_topics = n_topics, n_iter = 100)
  model.fit(X)
  distributions = model.components_
  doc_topic = model.doc_topic_
  def get_prob(i,j):
    sum = 0
    for topic in range(0, n_topics):
      sum += distributions[topic][j]*doc_topic[i][topic] 
    return sum

  display_results(name, get_prob)
  # num_labels = 10
  # name = "lda"
  # f_distributions = open('final10.beta', 'r')
  # f_labels = open('word-assignments10.dat', 'r')
  # all_labels = []
  # distributions = []
  # for l in f_distributions:
  #   distribution = l.split()
  #   distribution = [math.exp(float(i)) for i in distribution]
  #   #fill in 10 missing users at end
  #   for i in range(0, num_labels):
  #     distribution.append(0)
  #   distributions.append(distribution)


  # for l in f_labels:
  #   labels = [0] * num_labels
  #   l = l.split()
  #   for pair in l[1:]:
  #     pair = pair.split(":")
  #     topic = int(pair[1])
  #     labels[topic]+=1
  #   all_labels.append(labels)
  

  # f_labels.close()
  # f_distributions.close()

  # def get_prob(i,j):
  #   sum = 0
  #   count = 0
  #   for label in range(0, num_labels) :
  #     sum+= distributions[label][j]*all_labels[i][label]
  #     count+= all_labels[i][label]
  #   if count == 0:
  #    return 0
  #   return sum / count
  # display_results(name, get_prob)


def display_results(name, get_prob):
  f_test = open('testTriplets.txt', 'r')
  f_out = open(name+"Probs.txt", "w")
  test = np.array([l.split() for l in f_test.readlines()])
  probs = []
  trues = [] 
  for l in test: 
      i = int(l[0])
      j = int(l[1])
      value = int(l[2])

      prob = get_prob(i,j)
      if prob > 1 :
          prob = 1
      elif prob < 0:
          prob = 0
      f_out.write(str(prob)+"\n")
      trues.append(value)
      probs.append(prob)
  f_test.close()
  f_out.close()



  # fpr, tpr, thresholds = roc_curve(trues, probs, pos_label =1)
  # roc_auc = auc(fpr, tpr)
  # print "auc", roc_auc 
  # plt.figure()
  # plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
  # plt.plot([0, 1], [0, 1], 'k--')
  # plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.0])
  # plt.xlabel('False Positive Rate')
  # plt.ylabel('True Positive Rate')
  # plt.title(name + ' ROC curve')
  # plt.legend(loc="lower right")
  # plt.show()



#nmf()
#svd()
#kmeans()
lda()