import numpy as np

k=np.array([32.,23.,3.,2.,12.])
N=2
al=np.array([1,2,3,4,5])
w=np.cumsum(k)/sum(k)
r=np.random.random(N)
ind=np.zeros((N,len(k)))
cnt=np.zeros(N)
res=np.zeros(N)
for x in xrange(N):
	ind[x]=r[x]>w
	for a in ind[x]:
		if a:
			cnt[x]+=1
	res[x]=al[cnt[x]]
		
		
print 'w',w,'ind',ind,'cnt',cnt,'r',r,'res',res

