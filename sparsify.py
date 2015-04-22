from scipy import sparse
import numpy as np
from sklearn.decomposition import TruncatedSVD, RandomizedPCA, NMF
from sklearn.metrics import matthews_corrcoef, roc_auc_score, roc_curve, auc
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from functools import partial

f = open('txTripletsCounts.txt', 'r')
triplets = f.readlines()
triplets = np.array([l.split() for l in triplets])
row = [int(s) for s in triplets[:,0]]
col = [int(s) for s in triplets[:,1]]
data = [int(s) for s in triplets[:,2]]
bin_data = [1 for s in triplets[:,2]]
num_users = 444075
X = sparse.csc_matrix((bin_data, (row,col)), shape=(num_users, num_users))
f.close()
#cov = x * csc_matrix.transpose(x)


def nmf():
  name = "nmf"
  model = NMF(n_components = 25)
  print name
  model.fit(X)
  A_T = sparse.csc_matrix(model.components_)
  Z = A_T*X
  A = A_T.transpose()
  print "reconstruction error", model.reconstruction_err_
  def get_prob(i, j):
    return (A.getrow(i)*Z.getcol(j))[0,0]
  display_results(name, get_prob) 


def svd():
  name = "svd"
  model = TruncatedSVD(n_components = 25)
  print name
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
  model = KMeans()
  print name
  labels = model.fit_predict(X)
  def get_prob(i,j):
    return model.cluster_centers_[labels[i]][j]
  display_results(name, get_prob)



def display_results(name, get_prob):
  f_test = open('testTriplets.txt', 'r')
  test = np.array([l.split() for l in f_test.readlines()])
  tp = 0
  tps = []
  tn = 0
  tns = []
  fp = 0
  fps = []
  fn = 0
  fns = []
  count = 0
  cutoff = 1e-3
  bin_preds = []
  probs = []
  trues = [] 
  for l in test: 
      i = int(l[0])
      j = int(l[1])
      value = int(l[2])

      pred = get_prob(i,j)
      prob = pred
      if pred > 1 :
          prob = 1
      elif pred < 0:
          prob = 0

      if pred > cutoff: 
          bin_pred = 1
      else:
          bin_pred = 0

      if value == 0 and bin_pred == value:
          tn += 1
      if value == 1 and bin_pred == value:
          tp += 1
      if value == 0 and bin_pred != value:
          fp += 1
      if value == 1 and bin_pred != value:
          fn += 1

      trues.append(value)
      bin_preds.append(bin_pred)
      probs.append(prob)
  f_test.close()
  print "tp", tp
  print "tn", tn
  print "fp", fp
  print "fn", fn

  print "matthews correlation", matthews_corrcoef(trues, bin_preds)

  fpr, tpr, thresholds = roc_curve(trues, probs, pos_label =1)
  roc_auc = auc(fpr, tpr)
  print "auc", roc_auc 
  plt.figure()
  plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(name + ' ROC curve')
  plt.legend(loc="lower right")
  plt.show()



#nmf()
svd()
#kmeans()