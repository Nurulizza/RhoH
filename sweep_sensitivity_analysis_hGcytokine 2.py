from collections import OrderedDict
import numpy as np
import math
from SALib.sample import saltelli
from SALib.analyze import sobol
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, minimize,least_squares

import OpenCOR as oc

dict = {'geometry/dl4cytokines_parameter/k_f10': 0.0141382,
  'geometry/dl4cytokines_parameter/k_r10': 0.014043,
  'geometry/dl4cytokines_parameter/k_f1': 6.18038,
  'geometry/dl4cytokines_parameter/k_f5': 0.804035,
  'geometry/dl4cytokines_parameter/k_r1': 0.0014888,
  'geometry/dl4cytokines_parameter/k_r4': 0.13486,
  'geometry/dl4cytokines_parameter/k_r6': 0.71853,
  'geometry/dl4cytokines_parameter/k_f21': 0.0515956,
  'geometry/dl4cytokines_parameter/k_f22': 0.00297296,
  'geometry/dl4cytokines_parameter/k_f23': 0.00215999,
  'geometry/dl4cytokines_parameter/k_f24': 0.98438,
  'geometry/dl4cytokines_parameter/k_r21': 2.0772,
  'geometry/dl4cytokines_parameter/k_r23': 0.334514,  
  'hG_FC/k_f21': 0.016521,
  'hG_FC/k_r21': 0.0517365,
  'PI3K/k_f2': 8.91653,
  'PI3K/k_r2': 0.00611373,
  'PI3K/k_f3': 14.6231,
  'cytokines/k_f4': 0.0683688,
  'cytokines/k_f5': 23.1163,
  'cytokines/k_f32': 9.78575e-07,
  'cytokines/k_f31': 0.0291491,
  'dupont_NFAT/kf': 0.00647734}
  
varyme = 'geometry/dl4cytokines_parameter/k_r10'

# The state variable  or variables in the model that the data represents
num_series = 2
expt_state_uri = ['cytokines/IFN','cytokines/TNF']

#Some example output that we are maybe aiming for
times = np.array([0, 30, 60, 120, 240, 480, 960])*60
exp_data = np.zeros([num_series,len(times)])
exp_data[0,:] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.025, 1])*0.0001 #IFN
exp_data[1,:] = np.array([0.0,  0.0, 0.0, 0.17, 0.81, 0.74, 0.95])*0.0001 #TNF

#Number of samples to generate for each parameter
num_samples =  100

#Number of results to retain, if we store too many in high res parameter sweeps we can have memory issues
num_retain = 10

#List of parameters you want to exclude from fit
#fit_parameters_exclude = ['FCepsilonRI/pFC']

class Simulation(object):
    def __init__(self):
        self.simulation = oc.simulation()
        self.simulation.data().setStartingPoint(0)
        self.simulation.data().setEndingPoint(57600)
        self.simulation.data().setPointInterval(1)
        self.constants = self.simulation.data().constants()
        self.simulation.resetParameters()
        self.simulation.clearResults()
        for k,v in dict.items():
             self.constants[k]=v
        self.model_constants = OrderedDict({k: self.constants[k]
                                            for k in self.constants.keys()})  
		
    
    def run_once(self, c, v):
        self.simulation.resetParameters()
        self.constants[c] = v
        self.simulation.run()
        return (self.simulation.results().points().values(),
                self.simulation.results().states()['cytokines/IFN'].values())
    
    def run_sensitvity(self, c, scale=2.0):
        self.simulation.clearResults()
        v = self.model_constants[c]
        base = self.run_once(c, v)[1][times]
        divergence = 0.0
        for s in [1.0/scale, scale]:
            trial = self.run_once(c, s*v)[1][times]
            divergence += math.sqrt(np.sum((base - trial)**2))
        return divergence
    
    def evaluate_model(self, parameter_values):
        self.simulation.clearResults()
        for i, k in enumerate(self.model_constants.keys()):
            self.constants[k] = parameter_values[i]
        #print('Parameter set: ', parameter_values)
        self.simulation.run()
        return (self.simulation.results().states()['cytokines/IFN'].values()[times])
    
    def evaluate_ssq(self):
        self.simulation.clearResults()

        self.simulation.run()
        trial = np.zeros([num_series,len(times)])
        ssq = np.zeros(num_series+1)
		
        for i in range(0,num_series):
            trial[i,:] = self.simulation.results().states()[expt_state_uri[i]].values()[times]
            ssq[i+1] = math.sqrt(np.sum((exp_data[i,:]-trial[i,:])**2))
        ssq[0] = np.sum(ssq[1:num_series+1])
        return ssq 
        
    
    def run_parameter_sweep(self):
        num_cols = num_series + 1 + self.samples.shape[1]
        num_rows = num_retain+1
        Y = np.zeros([num_rows,num_cols])
        for i, X in enumerate(self.samples):
            ssq = self.evaluate_ssq(X)
            j = i
            if j < num_retain:
                Y[j,0] = ssq[0]
                for k in range(0,num_series):
                    Y[j,k+1] = ssq[k+1]
                Y[j,(k+2):num_cols]=X
            else:
                Y[num_retain,0] = ssq[0]
                for k in range(0,num_series):
                    Y[num_retain,k+1] = ssq[k+1]
                Y[num_retain,(k+2):num_cols]=X
                ind = np.argsort(Y[:,0])
                Y=Y[ind]
				
			#Want to retain top N here
        ind = np.argsort(Y[:,0])
        Z=Y[ind]
        return Z			


plt.close('all')

plt.semilogx([dict[varyme],dict[varyme]],[0,0.0005],label='fit value')
values = np.zeros([4,len(np.arange(-3,2,0.1))])
count = 0
for i in np.arange (-3,2,0.1):
      dict[varyme] = 10**i
      
      s = Simulation()
      
      v = s.evaluate_ssq()
      values[0,count] = 10**i
      values[1:4,count] = v
      count = count +1
      print(count,values[0,count-1])
print(values)
plt.semilogx(values[0,:],values[1,:],label='total error', color='black')
plt.semilogx(values[0,:],values[2,:],label='IFNgamma', color='blue')
plt.semilogx(values[0,:],values[3,:],label='TNFalpha', color='red')
plt.legend()
plt.xlabel(varyme)
plt.ylabel('Error metric')

plt.show()
