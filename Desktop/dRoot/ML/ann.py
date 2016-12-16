import numpy as np
import sys
class OneLayerBinaryNN:
	def __init__(self,X,y,hidUnits,epochs=20,lam=5*1e-3,learnRate=1e-3,bias=False):
		"""
		Pars - np.array(p) or list with p elements
 		X - np.array((m,p))
		y - np.array(m)
		"""
		assert(len(y) == X.shape[0]), 'Size of Y vector (%d) with must be consistent with given data matrix shape along axis 0 (%d)!'% (len(y) , X.shape[0])
		self.f = X.shape[1]
		self.X= X if bias == False else np.c_[np.ones((len(y),1)), X]
		self.y=y
		self.epochs = epochs
		self.hidUnits = hidUnits
		self.w1 = np.random.normal(size=(self.f+bias,hidUnits)) 
		self.w2 = np.random.normal(size=(hidUnits,1))
		self.b2 = np.random.normal(0.,0.5)
		#self.b1 = np.random.normal(0.,0.5,size=hidUnits)
		self.z2 = 0.
		#self.z2 = np.zeros((len(y),hidUnits))
		#self.z2 = np.zeros((len(y),1)
		self.CostVal = 0.
		self.lam = lam
		self.learnRate = learnRate
		self.PrevCostVal = 10.
		self.GradVal = 0.
		self._eps = 1e-6
		self.NormParams=np.zeros((2,self.f))
		self.bias = bias
		
	def normalize(self):
		for j in xrange(self.f):
			self.NormParams[0][j]=(self.X[:,j+self.bias]).mean()
			self.NormParams[1][j]=(self.X[:,j+self.bias]).std()

			self.X[:,j+self.bias]=((self.X[:,j+self.bias])-self.NormParams[0][j])/self.NormParams[1][j]
			
	def sigmoid(self,x):
		return 1./(1+np.e**(-1.*x))
	
	def fnc(self):

		self.z2 = np.dot(self.X,self.w1)
		a2 = self.sigmoid(self.z2)
		z3 = np.dot(a2,self.w2) + self.b2
		return a2,self.sigmoid(z3) # matrix of a2 and vector a3 
		
	def cost(self,h):
		
		self.CostVal= (-(np.dot(self.y,np.log(h)) + np.dot((1-self.y),np.log(1-h)))[0] + self.lam*((self.w1[:,1:]**2).sum() + np.dot(self.w2.T,self.w2)[0][0])/2)/len(self.y)
				
	def backprop(self,a3,a2):
		a1 = self.X
		y =self.y
		delta3 = a3	 - y.reshape((len(y),1))
		delta2 = np.dot(delta3,self.w2.T)*a2*(1.-a2)
		delta1 = np.dot(delta2,self.w1.T)*a1*(1.-a1) 
						
		#bias
		#print 'delta3.sh=',delta3.shape, '   delta.shape=',a3.shape

		deltab = np.dot(delta3.T,a3*(1-a3))				
						
		Delta2 = np.dot(delta3.T,a2)/len(y)
		Delta1 = np.dot(delta2.T,a1)/len(y)
		D2 = Delta2.T + self.lam*self.w2
		D1 = Delta1.T + self.lam*self.w1
		
		return D2,D1,deltab
	
	def updateWeights(self,D2,D1,deltab):
		self.w2 -= D2*self.learnRate
		self.w1 -= D1*self.learnRate
		self.b2 -= deltab*self.learnRate
		
	def clasifier(self):
		def clasifierFunc(X):
			for col in xrange(self.f):
				X[:,col] = (X[:,col]-self.NormParams[0,col])/self.NormParams[1,col]
			if self.bias:	
				X = np.c_[np.ones((len(X),1)), X]

			z2 = np.dot(X,self.w1)
			a2 = self.sigmoid(z2)
			z3 = np.dot(a2,self.w2) + self.b2
			return self.sigmoid(z3)
		return clasifierFunc	

	def train(self):
		self.normalize()
		i=0
		while abs(self.CostVal - self.PrevCostVal) > self._eps:
			i += 1
			self.PrevCostVal = self.CostVal
			#self.=0.+ self.CostVal
			a2,a3, = self.fnc()
			
			self.cost(a3)
			d2,d1,db2=self.backprop(a3,a2)
			
			self.updateWeights(d2,d1,db2)
			if i%1000 == 0:
				print 'J= ',self.CostVal, '  epoch = ',i  
				print 'J(',i,')-J(',i-1,')=',self.CostVal - self.PrevCostVal
				#print 'J(',i,'))=',self.CostVal
				print id(self.CostVal)==id(self.PrevCostVal)
		return self.clasifier()
	"""def Grad(self):
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
		
			 
def sigmoid(X):
	return 1./(1+np.e**(-1.*X))
		
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

"""
if __name__ == '__main__':
	data=np.loadtxt('KievFlats.txt')
	Y=data[:,0]-1
	x=data[:,1:]
	ann = OneLayerBinaryNN(x,Y,5)
	clasifier = ann.train()
	#ann.normalize()
	#print 'X normed =',ann.X
	#a2,a3 = ann.fnc(return_a2=True)
	#d2,d1=ann.backprop(Y,a3,a2)
	
