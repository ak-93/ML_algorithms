import numpy as np
import time
import sys
from matplotlib import pyplot as plt
class SVM:
	def __init__(self,X,y,C,eps,learningRate,mxVal):
		"""
		X - np.array((m,p))
		y - np.array(m)
		mxVal - maximum Value of cost considerable to minimize. If the value of cost function exceeds this MxVal,
		parameters will be changed to new random set
		eps - when eps exceeds the value of cost function iteration step, fit consider to be converged.
		"""
		assert(len(y) == X.shape[0]), 'Size of Y vector (%d) with must be consistent with given data matrix shape along axis 0 (%d)!'% (len(y) , X.shape[0])
		self.pars = np.random.normal(size=(len(X)+1))#ones(len(X)+1)
		self.X = X
		self.y = y
		self.C = C
		self.converged = False
		self._iter = 0
		self.FncVal = 1.
		self.PrevFncVal = 10.
		self.GradVal = np.zeros(len(X)+1)
		self._eps = eps
		self._n = len(X)
		self.learningRate = learningRate
		self.f = np.ones((len(X),len(X)+1))
		self.NormParams = np.zeros((2,(self.pars).size-1))
		self.MaxVal = mxVal
	def normalize(self):
		for j in xrange((self.X).shape[1]):
			self.NormParams[0][j]=(self.X[1:,j]).mean()
			self.NormParams[1][j]=(self.X[1:,j]).std()
			self.X[:,j]=((self.X[:,j])-self.NormParams[0][j])/self.NormParams[1][j]
	
	def transform(self,X):
		f = np.ones((len(X),len(self.X)+1))
		for j in xrange(self._n):
			f[:,j+1] = self.similarity(self.X[j],X)		
		return f
		
	def cost(self,z):
		f0=np.zeros_like(z)
		f1=np.zeros_like(z)

		f0[z+1>0]=z[z+1>0]+1
		f1[z-1<0]=-z[z-1<0]+1
		self.FncVal = self.C*((self.y*f1.T+(1-self.y)*f0.T)).sum(axis=1) + np.dot(self.pars[1:],self.pars[1:])/2.
		
	def similarity(self,x,y,sigma=1.):
		return np.exp(-(((x-y)**2).sum(axis=1))/2/(sigma*sigma))


	def Grad(self):
		self.GradVal = (self.C*(np.dot(self.y,self.f) - np.dot(1-self.y,self.f)))
		self.GradVal += np.r_[0,self.pars[1:]]
		self.GradVal /= (np.linalg.norm(self.GradVal))
		self.GradVal *= self.learningRate
		
		
	def train(self):
		t0 = time.time()
		self.normalize()
		self.f=self.transform(self.X)
		
		while not self.converged:
			if abs(self.PrevFncVal-self.FncVal)<self._eps:
				if self._iter>1 and self.FncVal<0.8*self.MaxVal:
					self.converged = True
			self._iter += 1
			if self._iter % 1000 == 0:
				self.PrevFncVal = self.FncVal
			z = np.dot(self.f,self.pars.reshape((len(self.pars),1)))
			self.cost(z)
			self.Grad()
			self.pars += self.GradVal.T

			
			if self._iter % 1000 == 0:
				print 'PreviousFncVal-FncVal',abs(self.PrevFncVal-self.FncVal)
				print 'iteration :', self._iter,'FuncVal = ',self.FncVal , 'Time since start: %.1f sec' %(time.time()-t0)
				
				if  self.PrevFncVal-self.FncVal < 0 or self.FncVal>self.MaxVal	:
					print 'Gradient descent failed after %d iterations!' %self._iter
					print '__________________________________________________'
					print 'Start minimizations with new initial parameters.\n\n\n'
					self.pars = np.random.normal(size=(len(self.X)+1))
					self._iter = 0
				

		print 'Fit has been convereged succesfully! The number of iterations: %d \nafter %.1f sec of minimization' %(self._iter,time.time()-t0)
		print 'Fnc value = ',self.FncVal
		
	def predictor(self):
		
		def model(x):
			X = np.copy(x)
			if X.shape[1]!=self.X.shape[1]:
				raise AttributeError ('Unappropriative shape of data matrix! Expected matrix with %d columns' %self.X.shape[1])
			for col in xrange(X.shape[1]):
				X[:,col] = (X[:,col]-self.NormParams[0,col])/self.NormParams[1,col]
			transformed = self.transform(X) 			#|  trans: n,f+1 ;  theta: f+1
			ans = (np.dot(transformed,self.pars)>0)+1
			return ans
		return model


if __name__ == '__main__':
	data=np.loadtxt('KievFlats.txt')
	Y=data[:,0]-1
	x=data[:,1:]
	model=SVM(x,Y,1000.,1e-1,1e-5,150e3)
	model.train()
	predictor = model.predictor()
	flats = np.array([[23.,34.,2014],[42.,67.,1932],[32.,43.,1965],[63.,85.,2001],[32.,56.,1989],[78.,103.,2012],[34.,38,1984],[78.,97,1989],[78.,97,2007],[29.,38,1992],[53.,78,2005],[42.,46,1959]])
	rooms=predictor(flats)
	print 'rooms:', rooms
	print 'in flats area(sq.m), cost($), built\n\t'
	print '\t','\n\t'.join([str(row) for row in flats])
