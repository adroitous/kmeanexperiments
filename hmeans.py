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


def split(X_train, centroid, labels, labelnum, ch_score_old, level):
	for e in labels:
		if e == labelnum:
			X_train.remove(label)
	km = TimeSeriesKMeans(n_clusters = 2,  metric = "euclidean", max_iter = 100, verbose = False, n_init = 10, random_state = seed).fit(X_train)
	centroid.remove(index(labelnum))
	centroid.extend(km.cluster_centers_)
	set(label)


def loaddata():
	datalist = UCR_UEA_datasets().list_datasets()
	r =[]
	for e in datalist:
		X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(e)
		if X_train is None:
			print(e)
			continue
		X_train = np.concatenate((X_train, X_test), 0)
		length = X_train.shape[0]
		result = []
		#min(int(math.ceil(length*0.5)), 100
		if X_train.shape[0]>4000:
			print(e)
			#r.append(kmeans(X_train, 2))
		#print(r)
	#np.savetxt("hkmeans:result", r)

def kmeans(X_train, n):
	seed = 0
	np.random.seed(seed)
	t0 = time.time()
	km = TimeSeriesKMeans(n_clusters = 10, metric = "euclidean", max_iter = 100, verbose = False, n_init = 10, random_state = seed).fit(X_train)
	labels = km.labels_
	#print (labels.shape)
	num = set(labels)
	#print(num)
	#print(X_train.shape)
	Xlist = []
	X_transform = transformdata(X_train, False)
	_ ,accuracyList = mainfunction(km.cluster_centers_, X_train)
	Alist = []
	for number in num:
		alist = []
		accuracy = []
		for index in range(len(X_transform)):
			if labels[index] == number:
				alist.append(X_transform[index])
				accuracy.append(accuracyList[index])
		Xlist.append(alist)
		Alist.append(sum(accuracy)/len(accuracy))
	i =0
	cluster_centers_ = km.cluster_centers_
	for e in Xlist:
		#print(len(e))
		if len(e) > 10 :
			e = np.asanyarray(e)
			km_Xlist = TimeSeriesKMeans(n_clusters = 2, metric = "euclidean", max_iter = 100, verbose = False, n_init = 10, random_state = seed).fit(e)
			if Alist[i] > evaluate_cluster(km_Xlist.cluster_centers_, e):
				cluster_centers_ = np.append(cluster_centers_,km_Xlist.cluster_centers_,axis = 0)
				#print("len(cluster_centers_)")
				#print(len(cluster_centers_))
				#mainfunction(cluster_centers_, X_train)
		i+=1
	a, _ = mainfunction(cluster_centers_, X_train)
	t1 = time.time()
	print(t1-t0)
	km = TimeSeriesKMeans(n_clusters = len(cluster_centers_), metric = "euclidean", max_iter = 100, verbose = False, n_init = 10, random_state = seed).fit(X_train)
	#print("original")
	b, _ = mainfunction(km.cluster_centers_, X_train)
	print(time.time()-t1)
	return (a-b)

def evaluate_cluster(cluster, X_train):
	D = transformdata(cluster, True)
	X = np.swapaxes(X_train,0,1)
	param = {'lambda1':0.15, 'numThreads':-1, 'mode':spams.PENALTY, 'L':20}
	(alpha, path) = spams.lasso(X, D=D, return_reg_path = True, **param)
	a = np.swapaxes(alpha.toarray(),0,1)
	reconstructed = []
	Dictionary = transformdata(cluster,False)
	X_transform = X_train
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
	return accuracy

def accuracylist(array):
	return [sum(e)/len(e) for e in array]
	

def mainfunction(cluster, X_train):
	param = {'lambda1':0.15, 'numThreads':-1, 'mode':spams.PENALTY, 'L':20}
	D = transformdata(cluster, True)
	X = transformdata(X_train, True)
	(alpha, path) = spams.lasso(X, D=D, return_reg_path = True, **param)
	a = np.swapaxes(alpha.toarray(),0,1)
	Dictionary = transformdata(cluster,False)
	X_transform = transformdata(X_train, False)
	'''
	ch_score = calinski_harabaz_score(X_transform, labels)
	db_score = davies_bouldin_score(X_transform,labels)
	sil_score = silhouette_score(X_transform,labels)
	'''
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
	accuracyList = accuracylist(accuracy)
	accuracy = reconstructed_error(accuracy)
	#t1 = time.time()
	#total = t1-t0
	#print([n,accuracy,ch_score,db_score,sil_score,total])
	#return [n,accuracy,ch_score,db_score,sil_score,total]
	#print(accuracy)
	return accuracy, accuracyList
training = loaddata()
