from collections import OrderedDict
import numpy as np
import math
from SALib.sample import saltelli
from SALib.analyze import sobol
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, minimize,least_squares

import OpenCOR as oc

bounds_dictionary = {'geometry/dl4cytokines_parameter/k_f10': [-3,2],
'geometry/dl4cytokines_parameter/k_r10': [-3,2], 'hG_FC/k_f21': [-3,2], 'hG_FC/k_r21': [-3,2], 'PI3K/k_f2': [-3,2], 
'PI3K/k_r2': [-3,2], 'PI3K/k_f3': [-3,2], 'cytokines/k_f4': [-5,5], 'cytokines/k_f5': [-1,3], 'cytokines/k_f32': [-10,-6],'cytokines/k_f31': [-3,2], 'geometry/dl4cytokines_parameter/k_f1': [0,2],
'geometry/dl4cytokines_parameter/k_f5': [-3,2], 'geometry/dl4cytokines_parameter/k_r1': [-3,0], 'geometry/dl4cytokines_parameter/k_r4': [-3,0], 'geometry/dl4cytokines_parameter/k_r6': [-3,0],
'dupont_NFAT/kf': [-3,2], 'geometry/dl4cytokines_parameter/k_f21': [-3,2], 'geometry/dl4cytokines_parameter/k_f22': [-3,2], 'geometry/dl4cytokines_parameter/k_f23': [-3,2], 'geometry/dl4cytokines_parameter/k_f24': [-3,2],
                     'geometry/dl4cytokines_parameter/k_r21': [-3,2], 'geometry/dl4cytokines_parameter/k_r23': [-3,2]}#'geometry/dl4cytokines_parameter/k_f5': [1,2],

# The state variable  or variables in the model that the data represents
num_series = 2
expt_state_uri = ['cytokines/IFN','cytokines/TNF']

#Some example output that we are maybe aiming for
times = np.array([0, 30, 60, 120, 240, 480, 960])*60#[0, 30, 60, 120, 240, 480, 960]
exp_data = np.zeros([num_series,len(times)])
exp_data[0,:] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.025, 1])*0.0001 #IFN
exp_data[1,:] = np.array([0.0,  0.0, 0.0, 0.17, 0.81, 0.74, 0.95])*0.0001 #TNF

#Number of samples to generate for each parameter
num_samples = 5

#Number of results to retain, if we store too many in high res parameter sweeps we can have memory issues
num_retain = 10

#List of parameters you want to exclude from fit
fit_parameters_exclude = ['V_PLC/V_plc', 'V_PLC/gamma', 'geometry/dl4cytokines_parameter/Ca_tot','geometry/dl4cytokines_parameter/alpha', 'geometry/dl4cytokines_parameter/b', 'geometry/dl4cytokines_parameter/Ccn',
    'geometry/dl4cytokines_parameter/gamma','geometry/dl4cytokines_parameter/hGdl4','geometry/dl4cytokines_parameter/k', 'geometry/dl4cytokines_parameter/K_1',
    'geometry/dl4cytokines_parameter/K_act', 'geometry/dl4cytokines_parameter/K_d', 'geometry/dl4cytokines_parameter/K_dN', 'geometry/dl4cytokines_parameter/k_f2', 
    'geometry/dl4cytokines_parameter/k_f3', 'geometry/dl4cytokines_parameter/k_f4', 'geometry/dl4cytokines_parameter/k_f6', 'geometry/dl4cytokines_parameter/k_f7',
    'geometry/dl4cytokines_parameter/K_inh', 'geometry/dl4cytokines_parameter/K_IP', 'geometry/dl4cytokines_parameter/K_k', 'geometry/dl4cytokines_parameter/K_minus', 
    'geometry/dl4cytokines_parameter/K_mN', 'geometry/dl4cytokines_parameter/K_p', 'geometry/dl4cytokines_parameter/K_p1', 'geometry/dl4cytokines_parameter/K_p2',
    'geometry/dl4cytokines_parameter/K_plus',  
    'geometry/dl4cytokines_parameter/M', 'geometry/dl4cytokines_parameter/n', 'geometry/dl4cytokines_parameter/Ntot',
    'geometry/dl4cytokines_parameter/n_a', 'geometry/dl4cytokines_parameter/n_d', 'geometry/dl4cytokines_parameter/n_i', 'geometry/dl4cytokines_parameter/n_p', 
    'geometry/dl4cytokines_parameter/Pi', 'geometry/dl4cytokines_parameter/stimEnd', 'geometry/dl4cytokines_parameter/V_k',                       
    'geometry/dl4cytokines_parameter/V_MP', 'geometry/dl4cytokines_parameter/V_p1',  'geometry/dl4cytokines_parameter/V_p2'] 
    #'geometry/dl4cytokines_parameter/k_f21', 'geometry/dl4cytokines_parameter/k_f22', 'geometry/dl4cytokines_parameter/k_f23', 'geometry/dl4cytokines_parameter/k_f24',
    #                 'geometry/dl4cytokines_parameter/k_r21', 'geometry/dl4cytokines_parameter/k_r23']


class Simulation(object):
    def __init__(self):
        self.simulation = oc.simulation()
        self.simulation.data().setStartingPoint(0)
        self.simulation.data().setEndingPoint(57600)
        self.simulation.data().setPointInterval(1)
        self.constants = self.simulation.data().constants()
        self.constant_parameter_names = sorted(list(self.constants.keys()))
        #print (self.constants)										
        for i in range(0,len(fit_parameters_exclude)):
            self.constant_parameter_names.remove(fit_parameters_exclude[i])
        
        self.model_constants = OrderedDict({k: self.constants[k]
                                            for k in self.constant_parameter_names})
	
        
        # default the parameter bounds to something sensible, needs to be set directly
        bounds = []
        for c in self.constant_parameter_names:
            v = self.constants[c];
            bounds.append([bounds_dictionary[c][0], bounds_dictionary[c][1]])
           
        # define our sensitivity analysis problem
        self.problem = {
                   'num_vars': len(self.constant_parameter_names),
                   'names': self.constant_parameter_names,
                   'bounds': bounds
                   }
        self.samples = saltelli.sample(self.problem, num_samples)
        #print(self.samples)
        np.savetxt("self.samples.txt", self.samples)
        
    def run_once(self, c, v):
        self.simulation.resetParameters()
        self.constants[c] = v
        self.simulation.run()
        return (self.simulation.results().points().values(),
                self.simulation.results().states()['cytokines/TNF'].values())
    
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
        for i, k in enumerate(self.constant_parameter_names):
            self.constants[k] = parameter_values[i]
        #print('Parameter set: ', parameter_values)
        self.simulation.run()
        return (self.simulation.results().states()['cytokines/TNF'].values()[times])
    
    def evaluate_ssq(self, parameter_values):
        self.simulation.clearResults()
        self.simulation.resetParameters()
		#This is not actually clearing and resetting results
        for i, k in enumerate(self.constant_parameter_names):
            self.constants[k] = 10.0**parameter_values[i]
            #print(k,self.constants[k])
        #print('Parameter set: ', self.constants)
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
                #print(Y)
				
	#Want to retain top N here
        ind = np.argsort(Y[:,0])
        Z=Y[ind]

        return Z

    def plot_n_best(self,n,param_sweep_results):
        for i in range(0,n):
            self.simulation.clearResults()
            self.simulation.resetParameters()
            for j, k in enumerate(self.constant_parameter_names):
                self.constants[k] = 10.0**param_sweep_results[i,j+num_series+1]
            #print(param_sweep_results[i,j+3])
            #print('Parameter set: ', self.constants)
            self.simulation.run()
            trial = np.zeros([num_series,len(times)])
            for i in range(0,num_series):
                trial[i,:] = self.simulation.results().states()[expt_state_uri[i]].values()[times]
            ax1.plot(times,trial[0,:])
            ax2.plot(times,trial[1,:])
        ax3.plot(param_sweep_results[0:n,1],param_sweep_results[0:n,2],"*")
    			
    def parameter_bounds(self):
        # # Set bounds for parameters (optional)
        parameter_bounds = [len(self.constant_parameter_names)*[0], len(self.constant_parameter_names)*[6]]
        for i in range(0,len(initial_params)):
            parameter_bounds[0][i] = 10**bounds_dictionary[self.constant_parameter_names[i]][0]
            parameter_bounds[1][i] = 10**bounds_dictionary[self.constant_parameter_names[i]][1]

        parameter_bounds = tuple(parameter_bounds)
        return parameter_bounds
			
    def model_function_lsq(self,params,times,exp_data, return_type, debug=False):
        if debug:
             print('Fitting Parameters:')
             print(params)

        self.simulation.resetParameters()
        self.simulation.clearResults()
        for j, k in enumerate(self.constant_parameter_names):
            #print(j,k,params[j])
            self.constants[k] = params[j]

        try:
            self.simulation.run()
        except RuntimeError:
            print("Runtime error:")
            for n, v in enumerate(params[0:len(self.constant_parameter_names)]):
                print('  {}: {}'.format(self.constant_parameter_names[n], v))
            raise

        if return_type == 'optimisation':
            f1 = self.simulation.results().states()[expt_state_uri[0]].values()[times]-exp_data[0,:]
            f2 = self.simulation.results().states()[expt_state_uri[1]].values()[times]-exp_data[1,:]
            f = np.concatenate((f1,f2))
            if debug:
                print('SSD:')
                print(sum(f**2))
        elif return_type == 'visualisation':
            f1 = self.simulation.results().states()[expt_state_uri[0]].values()[times]
            f2 = self.simulation.results().states()[expt_state_uri[1]].values()[times]
            f = np.vstack((f1,f2))
        return f

plt.close('all')

s = Simulation()

v = s.run_parameter_sweep()

initial_params = 10**v[0,num_series+1:len(v[0,:])]

print('Parameters estimated from sweep:')
for j, k in enumerate(s.constant_parameter_names):
    print('  {}: {:g} '.format(k, initial_params[j]))

parameter_bounds = s.parameter_bounds()


opt =least_squares(s.model_function_lsq, initial_params, args=(times,exp_data, 'optimisation'),
                               bounds=parameter_bounds,xtol=1e-8,verbose=1)

print('Parameters estimated from fit:')
for j, k in enumerate(s.constant_parameter_names):
    print('  {}: {:g}'.format(k, opt.x[j]))
	
f =s.model_function_lsq(opt.x, times, exp_data,'visualisation', debug=False)

fig, ax = plt.subplots()
plt.plot(times, exp_data[0,:], 'o', label='Experiment IFN', color='red')
plt.plot(times, f[0], '-', label='Model IFN', color='blue')
IFN_error = abs(f[0] - exp_data[0,:])

print('IFN error = ' + str(np.mean(IFN_error)) + ' (SD ' + str(np.std(IFN_error)) +')')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('IFNgamma (uM)')
fig.canvas.draw()
plt.show()

fig, ax = plt.subplots()
plt.plot(times, exp_data[1,:], 'o', label='Experiment TNF', color='red')
plt.plot(times, f[1], '-', label='Model TNF', color='blue')
TNF_error = abs(f[1] - exp_data[1,:])

print('TNF error = ' + str(np.mean(TNF_error)) + ' (SD ' + str(np.std(TNF_error)) +')')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('TNFalpha (uM)')
fig.canvas.draw()
plt.show()
