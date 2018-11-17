import numpy as np
from tslearn.datasets import UCR_UEA_datasets
import matplotlib.pyplot as plt

datalist = UCR_UEA_datasets().list_datasets()
ch =[]
si = []
dv = []
data = []
count = []
for e in datalist:
	try:
		n = np.loadtxt(e+ ":result")
	except:
		pass
	result = np.asanyarray(n)
	result = np.swapaxes(n,0,1)
	calinski_harabaz_score = np.corrcoef(result[1], result[2])
	silhouette_score = np.corrcoef(result[1], result[4])
	davies_bouldin_score = np.corrcoef(result[1],result[3])
	ch.append(abs(calinski_harabaz_score[0][1]))
	si.append(abs(silhouette_score[0][1]))
	dv.append(abs(davies_bouldin_score[0][1]))
	data.append(result[1])
	count.append(result[0])
ch = sum(ch)/len(ch)
si = sum(si)/len(si)
dv = sum(dv)/len(dv)	

data = np.asanyarray(data)
print(data)
#data = np.savetxt("result.csv", data, delimiter=",",fmt=('%f'))
fig = plt.figure()
'''
fig.suptitle('85 Datasets and Reconstruction Error', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
ax.set_ylabel('Reconstruction Quality (Euclidean distance from original)')
ax.set_xlabel('Clusters')

for x in range(len(count)):
	ax.plot(count[x], data[x])
plt.savefig('result.png')
plt.show()
'''
fig = plt.figure()
fig.suptitle('Archetypal Datasets', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
ax.set_ylabel('Reconstruction Quality (Euclidean distance from original)')
ax.set_xlabel('Clusters')
for x in range(0, len(count), 11):
	ax.plot(count[x], data[x])
plt.savefig('Archetypal.png')
plt.show()



print(ch)
print(si)
print(dv)