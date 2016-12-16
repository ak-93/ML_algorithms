import numpy as np

class nBayes:
	"""
	Assuming independance of features of events
	"""
	def __init__(self,X,y,bins):
		if len(X) != len(y):
			raise AttributeError("The size of data arrays are not concerted! Array X must have %d rows" %len(y))
		self.X = X
		self.y = y
		self.features = 1 if len(self.X.shape)==1 else self.X.shape[1]
		if self.features != len(bins):
			raise AttributeError("Unappropiative size of data and bines arrays! Expected bins array with %d elements" %self.features )
		self.classes = set(self.y)
		self.probClass = dict.fromkeys(self.classes,0)
		self.binsX = bins
		self.condProb=dict.fromkeys(self.classes)
		self.binnedX=np.zeros_like(X)
		"""#___________________________
		self.eVal = np.array([])
		self.eVectReduced = np.array([])
		self.eValReduced = np.array([])
		self.restored = np.array([])
		self.new = np.array([])
		self.var = np.array([])
		if len(self.X)
		self.NormParams=np.zeros((2,(self.X).shape[1]))
		#_______________________________"""
	def binning(self,X):
		binned = np.zeros_like(X)
		for j in xrange(len(self.binsX)):
			binned[:,j] = np.digitize(X[:,j],self.binsX[j])
		return binned
		
	def PrapareProb(self,Xbinned):
		for Class in self.classes:
			self.condProb[Class]=dict.fromkeys(range(self.features))
			for feature in self.condProb[Class]:
				self.condProb[Class][feature] = np.zeros_like(self.binsX[feature])
		for evt in xrange(len(self.y)):
			self.probClass[self.y[evt]] += 1.	
			for feature in xrange(self.features):
					self.condProb[self.y[evt]][feature][Xbinned[evt][feature]-1] += 1
		
	def norm(self):				
		for Class in self.classes:
			self.probClass[Class] /= len(y)
			for feature in xrange(self.features):
				self.condProb[Class][feature] /= sum(self.condProb[Class][feature]) 		
				"""for element in 	xrange(len(self.bins[feature])):
					self.condProb[Class][feature][element] = sum(Xbinned[:,element]==element+1)
				self.condProb[Class][feature] /= sum(self.condProb[Class][feature])"""
	def determineClass(self,X):
		sz = l if len(X.shape)==1 else X.shape[1]
		if sz != self.features:
			raise AttributeError("Unappropiative size of data array! Expected data array with %d elements, got %d" %(self.features,sz) )
		binned=self.binning(X)
		#binned=np.zeros_lile((sz,len(self.classes)))
		ans={}
		for evt in xrange(len(binned)): 
			ans[evt+1]={}
			denom = 0.
			for Class in self.classes:
				tmp = 1.
				for feature in xrange(self.features):
					tmp *= self.condProb[Class][feature][int(binned[evt][feature])-1]
				
				ans[evt+1][Class] = tmp*self.probClass[Class]
				denom += ans[evt+1][Class]	
			print denom
			print 'ans[evt+1][Class]','/','denom =',ans[evt+1],denom
			for Class in self.classes:
				ans[evt+1][Class] /= denom
				if np.isnan(ans[evt+1][Class]):
					ans[evt+1][Class] = 0.
		return ans		
	#def train(self):
		#for Class in self.y:
			
		
		
	#def classify(self,data):
		
if __name__=="__main__":
	data=np.loadtxt('/home/shurik/Desktop/KievFlats.txt')
	y,x = data[:,0],data[:,1:]
	#cost =  np.hstack((np.arange(20,51.,5.),np.arange(60,160,20.),200.,300.)))
	#yrs = np.hstack((np.array([1900.,1930.]),np.arange(1960,2021,10.)))
	#s = np.hstack((np.arange(20,91,5.),100.))
	
	yrs = np.hstack((np.array([1900.,1930.]),np.arange(1960,2011,10.)))
	s = np.hstack((np.arange(20,61,5.),70.,80.,90.))
	cost =  np.hstack((np.arange(20,51.,5.),60.,80.,100.,150.,200.))
	
	bns=(s,cost,yrs)
	mod=nBayes(x,y,bns)
	binned=mod.binning(X=mod.X)
	mod.PrapareProb(binned)
	mod.norm()
