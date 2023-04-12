def calc_entropys(X,labels):
  # calculate differential entropy of every cluster
  le=LabelEncoder()
  labels=le.fit_transform(labels)
  n_labels=len(le.classes_)
  entropys=[0]*n_labels
  for i in range(n_labels):
    cluster_i=X[labels==i]
    entropys[i]=calc_entropy(cluster_i)
  return entropys
def calc_entropy(X):
  mu=X.mean(axis=0) # mean
  sigma=np.cov(X.T) # covariance matrix
  mn_f=multivariate_normal(mu,sigma,
  allow_singular=True)
  return mn_f.entropy()
def calc_centroids(X,labels):
  # calculate cluster centers
  le = LabelEncoder()
  labels=le.fit_transform(labels)
  n_labels=len(le.classes_)
  centers={}
  for k in range(n_labels):
    centers[k]=(np.mean(X[labels==k],axis=0))
  return centers
def IP(X,labels):
  # claculate IP
  entropys=calc_entropys(X,labels)
  compactness=np.mean(entropys)
  centroids=list(calc_centroids(X,labels).values())
  seperation=calc_entropy(np.array(centroids))
  return compactness-seperation
