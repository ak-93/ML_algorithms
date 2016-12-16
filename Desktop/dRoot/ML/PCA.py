import numpy as np
from PIL import Image
class PCA:
	def __init__(self,X,thresh=0.1):
		self.X = X
		self.eVect = np.array([])
		self.eVal = np.array([])
		self.eVectReduced = np.array([])
		self.eValReduced = np.array([])
		self.restored = np.array([])
		self.new = np.array([])
		self.var = np.array([])
		self.NormParams=np.zeros((2,(self.X).shape[1]))
		#self._converged = False
		self.thresh = thresh
		#self._iter=0
		
	def normalize(self):
		for j in xrange((self.X).shape[1]):
			self.NormParams[0][j] = (self.X[:,j]).mean()
			self.NormParams[1][j] = (self.X[:,j]).std()
			self.X[:,j] = ((self.X[:,j])-self.NormParams[0][j])/self.NormParams[1][j]
		
		
	def Transform(self,returnNew=False):
		self.var = np.dot(self.X.T,self.X)/len(self.X)
		self.eVal,self.eVect = np.linalg.eig(self.var)
		#print self.eVal
		self.eVectReduced,self.eValReduced = self.eVect[:,self.eVal>self.thresh],self.eVal[self.eVal>self.thresh]
		self.new = np.dot(self.X,self.eVectReduced)
		if returnNew:
			return self.new 
	def Restore(self,returnRestored=False):
		self.restored = np.dot(self.new,self.eVectReduced.T)
		#for j in xrange((self.restored).shape[1]):
		#	self.restored[:,j] = self.restored[:,j] + self.NormParams[0,j]
		if returnRestored:
			return self.restored
	def compare(self):
		return self.X-self.restored


if __name__=="__main__":
	#x=np.loadtxt('/home/shurik/Desktop/KievFlats.txt')[:,1:]
	#x=np.random.poisson(120,(40,5	))
	#x=np.random.random((40,3))
	#print x
	img=Image.open("/home/shurik/Pictures/310720102140.jpg").convert('L')
	x = np.copy(np.asarray(img),dtype=float)
	md=PCA(x,.5)
	md.normalize()
	md.Transform()
	md.Restore()
	md.compare()
