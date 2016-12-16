import numpy as np
import sys
from matplotlib import pyplot as plt

class linMod:
	def __init__(self,X,y,Pars,alpha=1e-6):
		"""
		Pars - np.array(p) or list with p elements
 		X - np.array((m,p))
		y - np.array(m)
		"""
		#print X.shape
		#assert(len(Pars) == X.shape[1]), 'Size of vector (%d) with parameters must be consistent with given data shape along axis 1 (%d)!' %(len(Pars),X.shape[1])
		assert(len(y) == X.shape[0]), 'Size of Y vector (%d) with must be consistent with given data matrix shape along axis 0 (%d)!'% (len(y) , X.shape[0])
		if type(Pars)==list:
			Pars=np.append(np.array([]),Pars)
		Pars=Pars.reshape((len(Pars),1))	
		self.X=np.c_[np.ones((len(X),1)), X]
		self.y=y
		self.alpha = alpha
		self.pars = Pars
		self.parsPrev = 0.
		self.converged = False
		self._iter = 0
		self.FncVal = 100.
		self.PrevFncVal = 10.
		self.GradVal = 0.
		self._eps = 1e-6
		self._n = len(X)
		self.NormParams=np.zeros((2,(self.pars).size-1))

	def normalize(self):
		if len((self.X).shape)>1:
			for j in xrange(1,(self.X).shape[1]-1):
				self.NormParams[0][j]=(self.X[:,j]).mean()
				self.NormParams[1][j]=(self.X[:,j]).std()
				self.X[:,j]=((self.X[:,j])-self.NormParams[0][j])/self.NormParams[1][j]
		else:
			self.NormParams[0]=(self.X).mean()
			self.NormParams[1]=(self.X).std()
			self.X=((self.X)-self.NormParams[0])/self.NormParams[1]
			
		self.X=np.c_[np.ones((len(self.X),1)), self.X]

	def lin(self,x,p):
		print 'x.shape',x.shape
		print 'p.shape',p.shape
		return x.dot(p)

	def Fnc(self):
		prod=(self.X).dot(self.pars)
		self.FncVal = ((prod.T-self.y).dot((prod.T-self.y).T))/2./self._n #(-(self.y).dot(np.log(self.sigmoid(prod)))-(1-self.y).dot(np.log(1-self.sigmoid(prod))) )/self._n
	
	def Grad(self):
		prod = ((self.X).dot(self.pars))
		#print 'grad'
		#print 50*'-'
		#print 'sigmoid(aX)-y',self.sigmoid(prod)-self.y
		#print '(self.sigmoid(prod)-self.y).dot(self.X))',(self.sigmoid(prod)-self.y).dot(self.X)
		#print 50*'-'
		#print 'pars',self.pars
		#print 'prod.shape = ',prod.shape,'  y.shape = ',(self.y).shape,' X.shape = ',(self.X).shape
		#print '(prod-self.y)',(prod.T-self.y).shape 
		self.GradVal = -(prod.T-self.y).dot(self.X)/self._n
		#print self.GradVal
		self.GradVal /=(abs((self.GradVal)*self.pars.T))/self.alpha
		#print self.GradVal.shape
		#print 'grad',self.GradVal ,'    str 60'
	def _model(self,X,pars):
		def func(data=X,pars=self.pars):
			
			if len(data.shape) == 1:
				#assert (len(data) == len(pars)), 'Innappropriative data array size! Expected array with %d rows' %len(pars)
				data = data.reshape((len(data),1))
			else:
				assert(len(pars) == data.shape[1]), 'Size of vector (%d) with parameters must be consistent with given data shape along axis 1 (%d)!' %(len(pars),X.shape[1])
			data=np.c_[np.ones((len(data),1)), data]	
			for j in xrange(len(pars)):
				"""if j !=0 :
					data[:,j]=pars[j]*(data[:,j]-normPars[0][j])/normPars[1][j]
				else:"""
				#data[:,j]=pars[j]*data[:,j]
			return np.array(self.lin(data,pars))
		return func
		
	def fit(self):
		#self.normalize()
		while not self.converged:
			if abs(self.PrevFncVal-self.FncVal)<self._eps:
			#print 'delta= ',((self.PrevFncVal-self.FncVal)/self.FncVal)
			#if abs(self.pars-self.parsPrev)<self._eps:
				self.converged = True
			self._iter += 1
			self.PrevFncVal = self.FncVal
			self.Fnc()
			self.Grad()
			if self._iter % 1000 ==0:	
				print 'iteration :', self._iter,'FuncVal = ',self.FncVal
				print 'pars',self.pars.T,' grad =', self.GradVal
			#print '1-y',1-self.y, 'iteration :', self._iter
			#print self.pars, self.GradVal.T
			self.parsPrev = self.pars 
			self.pars += self.GradVal.T
			#if self._iter == 5:
			#	sys.exit()
		#pars = self.pars
		print 2*'\n',60*'*'
		print 'Fit has been convereged succesfully! The number of iterations: %d' %self._iter
		print 'Fnc value = ',self.FncVal
		print 60*'*',2*'\n'
		model = self._model(X=self.X,pars=self.pars)
		#return self.pars,self.NormParams
		print type(model)
		return model 


if __name__ == '__main__':
	m=700
	x=np.ones(m)
	Y=np.zeros(m)
	for k in xrange(m):
		x[k] = k
		Y[k] = np.random.poisson(10) + np.random.normal(0.001)*np.random.poisson(k)
	#data = np.loadtxt('lindata.txt')
	#Y=data[:,0]-1
	#x=data[:,1:]

	#fittedValues,dct=model.fitormal(30.)*np.random.poisson(k)
	#data = np.loadtxt('lindata.txt')
	#Y=data[:,0]-1
	#x=data[:,1:]
	params=np.array([15.,0.1])
	model=linMod(x,Y,params)
	#fittedValues,dct=model.fit()	
	modelFnc=model.fit()
