import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import UCR_UEA_datasets
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabaz_score
import spams
import math
import time

def transformdata(data, swapaxes = False):
	x = []
	for e in data:
		list =[]
		for a in e:
			list.append(a[0])
		x.append(list)
	x= np.asfortranarray(x)
	if swapaxes:
		x = np.swapaxes(x,0,1)
	x = np.asfortranarray(x/np.tile(np.sqrt((x*x).sum(axis = 0)), (x.shape[0],1)), dtype = float)
	return x

def reconstructed_error(data):
	list = []
	for e in data:
		list.append(sum(e)/len(e))
	return sum(list)/len(list)

def loaddata():
	datalist = UCR_UEA_datasets().list_datasets()
	print(datalist)
	for e in datalist:
		X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(e)
		if X_train is None:
			print(e)
			continue
		X_train = np.concatenate((X_train, X_test), 0)
		length = X_train.shape[0]
		result = []
		for x in range(2, min(int(math.ceil(length*0.5)), 100)):
			result.append(mainfunction(X_train,x))
		result = np.asanyarray(result)
		np.savetxt(e+ ":result", result)

def mainfunction(X_train, n):
	seed = 0
	np.random.seed(seed)
	t0 = time.time()
	km = TimeSeriesKMeans(n_clusters = n, metric = "euclidean", max_iter = 100, verbose = False, n_init = 10, random_state = seed).fit(X_train)
	labels = km.labels_
	param = {'lambda1':0.15, 'numThreads':-1, 'mode':spams.PENALTY, 'L':20}
	D = transformdata(km.cluster_centers_, True)
	X = transformdata(X_train, True)
	(alpha, path) = spams.lasso(X, D=D, return_reg_path = True, **param)
	a = np.swapaxes(alpha.toarray(),0,1)
	Dictionary = transformdata(km.cluster_centers_,False)
	X_transform = transformdata(X_train, False)
	ch_score = calinski_harabaz_score(X_transform, labels)
	db_score = davies_bouldin_score(X_transform,labels)
	sil_score = silhouette_score(X_transform,labels)
	reconstructed = []
	for e in a:
		final = [np.zeros(X_transform.shape[1])]
		for index, element in enumerate(e):
			if element != 0:
				list = element*Dictionary[index]
				final.append(list)
		final = sum(final)
		final = np.asfortranarray(final)
		reconstructed.append(final)
	accuracy = np.subtract(X_transform, reconstructed)
	accuracy = reconstructed_error(accuracy)
	t1 = time.time()
	total = t1-t0
	print([n,accuracy,ch_score,db_score,sil_score,total])
	return [n,accuracy,ch_score,db_score,sil_score,total]
training = loaddata()
