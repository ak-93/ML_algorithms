import numpy as np
import sys
class logMod:
	def __init__(self,X,y,Pars):
		"""
		Pars - np.array(p) or list with p elements
 		X - np.array((m,p))
		y - np.array(m)
		"""
		assert(len(Pars) == X.shape[1]), 'Size of vector (%d) with parameters must be consistent with given data shape along axis 1 (%d)!' %(len(Pars),X.shape[1])
		assert(len(y) == X.shape[0]), 'Size of Y vector (%d) with must be consistent with given data matrix shape along axis 0 (%d)!'% (len(y) , X.shape[0])
		if type(Pars)==list:
			Pars=np.append(np.array([]),Pars)
		Pars=Pars.reshape((len(Pars),1))	
		self.X=X
		self.y=y
		self.pars = Pars
		self.converged = False
		self._iter = 0
		self.FncVal = 0
		self.PrevFncVal = 10.
		self.GradVal = 0
		self._eps = 1e-6
		self._n = len(X)
		self.NormParams=np.zeros((2,(self.pars).size))
	def normalize(self):
		for j in xrange((self.X).shape[1]):
			self.NormParams[0][j]=(self.X[:,j]).mean()
			self.NormParams[1][j]=(self.X[:,j]).std()

			self.X[:,j]=((self.X[:,j])-self.NormParams[0][j])/self.NormParams[1][j]
		
	def sigmoid(self,x):
		return 1./(1+np.e**(-1.*x))

	def Fnc(self):
		prod=(self.X).dot(self.pars)
		print 'prod',prod
		print 'np.log(1-self.sigmoid(prod))',np.log(1-self.sigmoid(prod))
	
		self.FncVal = (-(self.y).dot(np.log(self.sigmoid(prod)))-(1-self.y).dot(np.log(1-self.sigmoid(prod))) )/self._n
	def Grad(self):
		prod = ((self.X).dot(self.pars)).T
		#print 'grad'
		#print 50*'-'
		#print 'sigmoid(aX)-y',self.sigmoid(prod)-self.y
		#print '(self.sigmoid(prod)-self.y).dot(self.X))',(self.sigmoid(prod)-self.y).dot(self.X)
		#print 50*'-'
		
		self.GradVal= ((self.sigmoid(prod)-self.y).dot(self.X))/self._n
		
	def _model(self,X,normPars,pars):
		def func(data=X):
			
			if len(data.shape) == 1:
				assert (len(data) == len(pars)), 'Innappropriative data array size! Expected array with %d rows' %len(pars)
				data = data.reshape((1,len(data)))
			else:
				assert(len(pars) == data.shape[1]), 'Size of vector (%d) with parameters must be consistent with given data shape along axis 1 (%d)!' %(len(Pars),X.shape[1])
			
			for j in xrange(len	(pars)):
				data[:,j]=pars[j]*(data[:,j]-normPars[0][j])/normPars[1][j]
			
			return np.array((self.sigmoid(data.sum(axis=1))).round()+1,dtype=int)
		return func
	def fit(self):
		self.normalize()
		while not self.converged:
			if abs(self.PrevFncVal-self.FncVal)<self._eps:
				self.converged = True
			self._iter += 1
			self.PrevFncVal = self.FncVal
			self.Fnc()
			self.Grad()
			
			print 'iteration :', self._iter,'FuncVal = ',self.FncVal
			print 'pars',self.pars,' grad =', self.GradVal
		#print '1-y',1-self.y, 'iteration :', self._iter
			#print self.pars, self.GradVal.T
			self.pars -= self.GradVal.T
			#if self._iter == 3:
			#	sys.exit()
		#pars = self.pars
		print 'Fit has been convereged succesfully! The number of iterations: %d' %self._iter
		print 'Fnc value = ',self.FncVal
		model = self._model(X=self.X,normPars=self.NormParams,pars=self.pars)
		#return self.pars,self.NormParams
		print type(model)
		return model 
		
			 
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
	params=np.array([.1,.3,0.01])
	model=logMod(x,Y,params)
	#fittedValues,dct=model.fit()	
	modelFnc=model.fit()	
