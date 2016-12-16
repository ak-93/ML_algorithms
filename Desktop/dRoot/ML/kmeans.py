import numpy as np

class Kmeans:
	def __init__(self,X,K=2,eps=1e-5,centrInits=5 ,mu=np.array([])):
		self.X = X
		self.features = 1 if len(X.shape)==1 else X.shape[1]
		self.clustersAssigned = -np.ones(len(X),dtype=int)
		self._iter = 0
		
		self.costM = np.zeros((centrInits,K+1))
		#self.coor((K,features))
		#self.costGrad = 0.
		#if len(mu)>0:
		#	self.K = len(mu)
		#	self.mu = mu
		#else:
		self.Val = 0.
		self.PrevVal = 0.
		
		self.K = K
		self.mu = np.zeros((K, self.features))
		self.muPrev = np.zeros_like(self.mu) 
		self.eps = eps*np.ones(K)
		
		#print '___________________________________'
		#print '__init__ method '
		#print 'muPrev.shape',(self.muPrev).shape,'  mu.shape=',(self.mu).shape,'  eps.shape=',(self.eps).shape 
		#print '___________________________________'
		#print ''


	def Cost(self):
	#	self.costVal =  sum(((self.X-(self.mu[self.clusterAssigned]).reshape((len(X),K)))**2).sum(axis=1))/len(X)
		#print 'X-mu',((self.X-(self.mu[self.clustersAssigned]).reshape((len(self.X),self.features)))**2)
		return sum(((self.X-(self.mu[self.clustersAssigned]).reshape((len(self.X),self.features)))**2).sum(axis=1))/len(self.X)
	
	def rand(self,beginInd,endInd,number):
		r=np.random.randint(beginInd,endInd,number)
		if len(set(r)) == len(r):
			return r
		else:
			self.rand(beginInd,endInd,number)
			
	def findClosest(self):
		tmp = np.zeros((len(self.X),self.K))
		#nearest = np.zeros((len(self.X),dtype=int)
		for j in xrange(self.K):
			tmp[:,j] = ((self.X-self.mu[j])**2).sum(axis=1)
		for i in xrange(len(self.X)):
			self.clustersAssigned[i] = np.argmin(tmp[i])
	
	def initCentroids(self):
		for j in xrange(len(self.costM)):
			vectors = self.rand(0,len(self.X),self.K)
			self.mu=self.X[vectors]
			self.findClosest()
			self.costM[j,0] = self.Cost()	
			self.costM[j,1:] = vectors
		self.Val=self.costM[np.argmin(self.costM[:,0])][0]
		bestVectors=self.costM[np.argmin(self.costM[:,0])][1:]
		self.mu = self.X[bestVectors.astype(int)]
		
	def calcCentroids(self):
		self.muPrev=np.copy(self.mu)
		for c in xrange(self.K):
			self.mu[c] = (self.X[self.clustersAssigned == c]).mean(axis=0) 
		
	def learn(self):
		self.initCentroids() 
		converged = False
		#print 'muPrev.shape',self.muPrev.shape,'  mu.shape=',self.mu.shape,'  eps.shape=',self.eps
	
		
		print 'val=',self.Val,' previous=',self.PrevVal
		#while abs(self.Val - self.PrevVal)>1e-15:
		while (((self.muPrev-self.mu)**2).sum(axis=1)>self.eps).all():
			self.findClosest()
			
			self.PrevVal = self.Val
			self.calcCentroids()
			self.Val = self.Cost()
			self._iter += 1 
			#if self._iter%100==0:
			print ' '
			print 'iteration %d' %self._iter
			print 'val=',self.Val,' previous=',self.PrevVal
			
			#print '|mu-muPrev|^2',(((self.muPrev-self.mu)**2).sum(axis=1)>self.eps).all()
			
			
			#print 'X=',self.X
			print ' '
if __name__ == '__main__':
	d = np.zeros((500,3))
	d[:200,1]=np.random.poisson(17,200)
	d[200:,1]=np.random.poisson(7,300)
	d[200:,0]=np.random.poisson(12,300)
	d[:200,0]=np.random.poisson(2,200)
	d[200:,2]=np.random.poisson(10,300)
	d[:200,2]=np.random.poisson(10,200)

	kmn=Kmeans(d,K=5,)
	kmn.learn()
