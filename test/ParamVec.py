#
# A small utility class to deal with parameter vectors
#
import numpy as np

class ParamVec:
    def __init__(self):
        self._names=[]
        self._latexnames=[]
        self._values=[]
        self._minvalues=[]
        self._maxvalues=[]

    def addParam(self,name,value,latexname=None,min=-1e30,max=1e30):
        self._names.append(name)
        self._values.append(value)
        if latexname is None:
            slatexname=name
        self._latexnames.append(latexname)
        self._minvalues.append(min)
        self._maxvalues.append(max)

    def setValues(self,values):
        self._values=values
        
    def index(self,name):
        return self._names.index(name)

    def value(self,name):
        return self._values[self.index(name)]

    def latexName(self,name):
        return self._latexnames[self.index(name)]

    def values(self):
        return np.array(self._values)

    def bounds(self):
        return list(zip(self._minvalues,self._maxvalues))
    
    
    def __len__(self):
        return len(self._names)
