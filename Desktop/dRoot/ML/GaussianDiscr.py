
import numpy as np
import sys
class discrimGauss:
	def __init__(self,X,y):
		"""
 		X - np.array((m,p))
		y - np.array(m)
		"""
		assert(len(y) == X.shape[0]), 'Size of Y vector (%d) with must be consistent with given data matrix shape along axis 0 (%d)!'% (len(y) , X.shape[0])
		self.X=X
		self.y=y
		self.f = X.shape[1]
		self.p = float((y==0).sum())/len(y)
		self.mu0 = np.zeros(self.f)
		self.mu1 = np.zeros(self.f)
		self.Sigma = np.zeros((self.f,self.f))
		self.NormParams=np.zeros((2,self.f))
	
	def normalize(self):
		for j in xrange(self.f):
			self.NormParams[0][j]=(self.X[:,j]).mean()
			self.NormParams[1][j]=(self.X[:,j]).std()
			self.X[:,j]=((self.X[:,j])-self.NormParams[0][j])/self.NormParams[1][j]
		
		
	def calcParameters(self):
		self.mu0 = (self.X[self.y==0]).mean(axis=0)
		self.mu1 = self.X[self.y==1].mean(axis=0)
		delta0 = (self.X[self.y==0]-self.mu0)
		delta1 = (self.X[self.y==1]-self.mu1)
		self.Sigma = (np.dot(delta0.T,delta0)+np.dot(delta1.T,delta1))/len(self.y) 

	def clasifier(self):
		self.normalize()
		self.calcParameters()
		
		def func(X):
			x = np.copy(X)
			if len(X.shape)==1:
				x = x.reshape((len(X),1))
			for j in xrange(self.f):
				x[:,j]=((X[:,j])-self.NormParams[0][j])/self.NormParams[1][j]

			pXy0 = np.exp(-(((x-self.mu0).dot(np.linalg.inv(self.Sigma)))*((x-self.mu0))).sum(axis=1)/2.)/(abs(np.linalg.det(self.Sigma))*(2*np.pi)**float(self.f))**0.5
			pXy1 = np.exp(-(((x-self.mu1).dot(np.linalg.inv(self.Sigma)))*((x-self.mu1))).sum(axis=1)/2.)/(abs(np.linalg.det(self.Sigma))*(2*np.pi)**float(self.f))**0.5
			
			y = 1./(1. + pXy1*(1.-self.p)/(pXy0*self.p))
			rooms=[]
			for prob in y:
				if prob>0.5:
					rooms.append('1 room ('+str(prob)[:4]+' probability)')
				else:
					rooms.append('2 room ('+str(1-prob)[:4]+' probability)')
			return rooms
		return func			

if __name__ == '__main__':
	data=np.loadtxt('KievFlats.txt')
	Y=data[:,0]-1
	x=data[:,1:	]
	GaussianAnalysis=discrimGauss(x,Y)
	predictor = GaussianAnalysis.clasifier()
	X = np.array([[34.,39.,1978],[43.,57.,2007],[38.,44.,1989],[68.,72.,1974]])
	predictor(X)
