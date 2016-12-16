import numpy as np
import sys
from matplotlib import pyplot as plt
class SVM:
	def __init__(self,X,y,C):
		"""
		Pars - np.array(p) or list with p elements
 		X - np.array((m,p))
		y - np.array(m)
		"""
		#assert(len(Pars) == X.shape[1]+1), 'Size of vector (%d) with parameters must be consistent with given data shape along axis 1 (%d)!' %(len(Pars),X.shape[1])
		assert(len(y) == X.shape[0]), 'Size of Y vector (%d) with must be consistent with given data matrix shape along axis 0 (%d)!'% (len(y) , X.shape[0])
			
		#if len(Pars) == len(X)+1:
		#	if type(Pars)==list:
		#		Pars=np.append(np.array([]),Pars)
		#	Pars=Pars.reshape((len(Pars),1))
		self.pars = np.random.normal(size=(len(X)+1))#ones(len(X)+1)
		#self.pars[0]=1.	
		self.X = X
		self.y = y
		self.C = C
		self.converged = False
		self._iter = 0
		self.FncVal = 1.
		self.PrevFncVal = 10.
		self.GradVal = np.zeros(len(X)+1)
		self._eps = 1e-10
		self._n = len(X)
		self.f = np.ones((len(X),len(X)+1))
		self.NormParams = np.zeros((2,(self.pars).size-1))
		
	def normalize(self):
		for j in xrange((self.X).shape[1]):
			self.NormParams[0][j]=(self.X[1:,j]).mean()
			self.NormParams[1][j]=(self.X[1:,j]).std()
			self.X[:,j]=((self.X[:,j])-self.NormParams[0][j])/self.NormParams[1][j]
	
	def transform(self,X):
		f = np.ones((len(X),len(X)+1))
		for j in xrange(len(X)):
			f[:,j+1] = self.similarity(X,X[j])		
		return f
		
	def cost(self,z):
		# z = dot(f,theta)
		f0=np.zeros_like(z)
		f1=np.zeros_like(z)

		f0[z+1>0]=z[z+1>0]+1
		f1[z-1<0]=-z[z-1<0]+1
		#f0[z+1>0]=z[z+1>0]+1
		#f1[z-1<0]=-z[z-1<0]+1
		#print 'y*f1=',(self.y*f1.T).shape
		#print '(1-y)*f0=',((1-self.y)*f0.T).shape
		#print 'y*f1+(1-y)*f0=',((self.y*f1.T+(1-self.y)*f0.T)).shape
		#print 'sum(y*f1+(1-y)*f0))=',((self.y*f1.T+(1-self.y)*f0.T).sum(axis=1))
		self.FncVal = self.C*((self.y*f1.T+(1-self.y)*f0.T)).sum(axis=1) + np.dot(self.pars[1:],self.pars[1:])/2.
		
	def similarity(self,x,y,sigma=.5):
		return np.exp(-(((x-y)**2).sum(axis=1))/2/(sigma*sigma))


	def Grad(self):
		
		ind = self.y*np.dot(self.f,self.pars).T
		#print 'ind',ind
		self.GradVal[:] = 0.
		#print ind<1.
		#print 'self.y.reshape((len(self.y),1))*self.f',np.dot(self.y,self.f).shape
		self.GradVal[ind < 1.] = -np.dot(self.y,self.f)[ind < 1.]/self.pars[ind < 1.]/1e4
		#((self.sigmoid(prod)-self.y).dot(self.X))/self._n
		#print 'grad=',self.GradVal	
	"""def _model(self,X,normPars,pars):
		
		def func(data=X):
			for j in xrange(X.shape[1]):
				data[:,j]=((data[:,j])-self.NormParams[0][j])/self.NormParams[1][j]
			f = self.transform(data)
			
			if len(data.shape) == 1:
				assert (len(data) == len(pars)), 'Innappropriative data array size! Expected array with %d rows' %len(pars)
				data = data.reshape((1,len(data)))
			else:
				assert(len(pars) == data.shape[1]), 'Size of vector (%d) with parameters must be consistent with given data shape along axis 1 (%d)!' %(len(Pars),X.shape[1])
			
			return np.array((self.sigmoid(data.sum(axis=1))).round()+1,dtype=int)
		return func"""
	def train(self):
		self.normalize()
		self.f=self.transform(self.X)
		while not self.converged:
			if abs((self.PrevFncVal-self.FncVal)/self.FncVal)<self._eps:
				self.converged = True
			self._iter += 1
			self.PrevFncVal = self.FncVal
			z = np.dot(self.f,self.pars.reshape((len(self.pars),1)))
			self.cost(z)
			self.Grad()
			if self._iter % 1000 == 0:
				print 'self.PrevFncVal-self.FncVal',abs(self.PrevFncVal-self.FncVal)

				print 'iteration :', self._iter,'FuncVal = ',self.FncVal
				print 'pars',self.pars,' grad =', self.GradVal
			#print '1-y',1-self.y, 'iteration :', self._iter
			#print self.pars, self.GradVal.T
			self.pars += self.GradVal.T
			
			#if self._iter == 3:
			#	sys.exit()
		#pars = self.pars
		print 'Fit has been convereged succesfully! The number of iterations: %d' %self._iter
		print 'Fnc value = ',self.FncVal
		#model = self._model(X=self.X,normPars=self.NormParams,pars=self.pars)
		#return self.pars,self.NormParams
		#print type(model)
		#return model 
		
					
 
"""def sigmoid(X):
	return 1./(1+np.e**(-1.*X))
		
def compFnc(Pars,X,y ):
	
	assert(len(Pars) == X.shape[1]), 'Size of vector (%d) with parameters must be consistent with given data shape along axis 1 (%d)!', %(len(Pars),X.shape[1])
	assert(len(y) == X.shape[0]), 'Size of Y vector (%d) with must be consistent with given data matrix shape along axis 0 (%d)!',% (len(y) , X.shape[0])
	#if type(pars)==list:
	#	pars=np.append(np.array([]),Pars)
	#else:
	#	pars=Pars
	pars=pars.reshape((len(pars),1))
	prod=X.dot(pars)
	
	Fnc=-y.dot(np.log(sigmoid(prod)))-(1-y).dot(np.log(1-sigmoid(prod)))
	return Fnc

def compGrad(X,y,pars):
	
	pars=pars.reshape((len(pars),1))
	prod=(X.dot(pars)).T
	grad=(prod-y).dot(X)
"""
if __name__ == '__main__':
	data=np.loadtxt('KievFlats.txt')
	Y=data[:,0]-1
	x=data[:,1:]
	#params=np.array([.001,.003,.0001])
	#params=np.array([.01,.03,0.001])
	#params=np.array([.1,.3,0.01])
	model=SVM(x,Y,1000.)
	#fittedValues,dct=model.fit()	
	modelFnc=model.train()	

