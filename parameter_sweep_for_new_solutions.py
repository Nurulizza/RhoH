from collections import OrderedDict
import numpy as np
import math
from SALib.sample import saltelli
from SALib.analyze import sobol
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, minimize,least_squares

import OpenCOR as oc

run_from = 1

#List the unknown parameters and ranges over which you want to assess the model
bounds_dictionary ={'geometry/dl4cytokines_parameter/k_f10': [-3,2], 'geometry/dl4cytokines_parameter/k_r10': [-3,2], 'hG_FC/k_f21': [-3,2], 'hG_FC/k_r21': [-3,2], 'PI3K/k_f2': [-3,2], 
                    'PI3K/k_r2': [-3,2], 'PI3K/k_f3': [-3,2], 'cytokines/k_f4': [-5,5], 'cytokines/k_f5': [-1,3], 'cytokines/k_f32': [-10,-6],'cytokines/k_f31': [-3,2],
                    'geometry/dl4cytokines_parameter/k_f1': [0,2], 'geometry/dl4cytokines_parameter/k_f5': [-3,2], 'geometry/dl4cytokines_parameter/k_r1': [-3,0],
                    'geometry/dl4cytokines_parameter/k_r4': [-3,0], 'geometry/dl4cytokines_parameter/k_r6': [-3,0], 'dupont_NFAT/kf': [-3,2], 'geometry/dl4cytokines_parameter/k_f21': [-3,2],
                    'geometry/dl4cytokines_parameter/k_f22': [-3,2], 'geometry/dl4cytokines_parameter/k_f23': [-3,2], 'geometry/dl4cytokines_parameter/k_f24': [-3,2],
                    'geometry/dl4cytokines_parameter/k_r21': [-3,2], 'geometry/dl4cytokines_parameter/k_r23': [-3,2]}



#List of parameters you want to exclude from fit
fit_parameters_exclude =['V_PLC/V_plc', 'V_PLC/gamma', 'geometry/dl4cytokines_parameter/Ca_tot','geometry/dl4cytokines_parameter/alpha', 'geometry/dl4cytokines_parameter/b',
                         'geometry/dl4cytokines_parameter/Ccn', 'geometry/dl4cytokines_parameter/gamma','geometry/dl4cytokines_parameter/hGdl4','geometry/dl4cytokines_parameter/k',
                         'geometry/dl4cytokines_parameter/K_1', 'geometry/dl4cytokines_parameter/K_act', 'geometry/dl4cytokines_parameter/K_d', 'geometry/dl4cytokines_parameter/K_dN',
                         'geometry/dl4cytokines_parameter/k_f2', 'geometry/dl4cytokines_parameter/k_f3', 'geometry/dl4cytokines_parameter/k_f4', 'geometry/dl4cytokines_parameter/k_f6',
                         'geometry/dl4cytokines_parameter/k_f7', 'geometry/dl4cytokines_parameter/K_inh', 'geometry/dl4cytokines_parameter/K_IP', 'geometry/dl4cytokines_parameter/K_k',
                         'geometry/dl4cytokines_parameter/K_minus', 'geometry/dl4cytokines_parameter/K_mN', 'geometry/dl4cytokines_parameter/K_p', 'geometry/dl4cytokines_parameter/K_p1',
                         'geometry/dl4cytokines_parameter/K_p2', 'geometry/dl4cytokines_parameter/K_plus', 'geometry/dl4cytokines_parameter/M', 'geometry/dl4cytokines_parameter/n',
                         'geometry/dl4cytokines_parameter/Ntot', 'geometry/dl4cytokines_parameter/n_a', 'geometry/dl4cytokines_parameter/n_d', 'geometry/dl4cytokines_parameter/n_i',
                         'geometry/dl4cytokines_parameter/n_p', 'geometry/dl4cytokines_parameter/Pi', 'geometry/dl4cytokines_parameter/stimEnd', 'geometry/dl4cytokines_parameter/V_k',                       
                         'geometry/dl4cytokines_parameter/V_MP', 'geometry/dl4cytokines_parameter/V_p1',  'geometry/dl4cytokines_parameter/V_p2'] 

                         
# The state variable  or variables in the model that the data represents
num_series = 2
expt_state_uri = ['cytokines/IFN','cytokines/TNF']

# Define the target values for each of the three sets of experimental data
mean_IFN = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.025, 1])*0.0001
sigma_IFN = np.array([0.0000022, 0.0000022, 0.0000022, 0.0000022, 0.0000022, 0.0000022, 0.0000022]) #Note this will be an actual value from the experimental data paper 

mean_TNF = np.array([0.0,  0.0, 0.0, 0.17, 0.81, 0.74, 0.95])*0.0001
sigma_TNF = np.array([0.0000011, 0.0000011, 0.0000011, 0.0000011, 0.0000011, 0.0000011, 0.0000011]) #Note this will be an actual value from the experimental data paper 

exp_data = np.array([[mean_IFN,sigma_IFN],[mean_TNF,sigma_TNF]])
#print(mean_IFN)
#print(sigma_IFN)
#print(mean_TNF)
#print(sigma_TNF)
print(exp_data)
#print(exp_data[0,:])
#print(exp_data[1,:])

#Duration of each simulation
start_time = 0.0 #simulation start time
end_time = 57600.0 #simulation end time
point_interval = 5
time_export = 100 #we wont want to export at all timesteps as files will be big so we might look to exporting every 10 or 100  timesteps
times = np.linspace(start_time,end_time,int(end_time/point_interval))
#Define an export directory - this will generate a lot of files so you might want to export to a hard drive
export_directory = './' #e.g. '/harddrivename/foldername/'
export_parameter_filename = 'parameters_id_'
export_solution_filename = 'solution_id_'

#Number of samples to generate for each parameter
# You do want this to be relatively high
num_samples =  1


class Simulation(object):
    def __init__(self):
        self.simulation = oc.simulation()
        self.simulation.data().setStartingPoint(start_time)
        self.simulation.data().setEndingPoint(end_time)
        self.simulation.data().setPointInterval(point_interval)
        self.constants = self.simulation.data().constants()
        self.constant_parameter_names = sorted(list(self.constants.keys()))
        
                                            
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
        print('Total number of simulations to be run: ',len(self.samples))
        
    def run_once(self):
        self.simulation.resetParameters()
        self.simulation.run()
        end_time_solution = np.zeros(num_series)
        for i in range(0,num_series):
            end_time_solution[i] = self.simulation.results().states()[expt_state_uri[i]].values()[-1] #Last entry for each simulation result
        print(end_time_solution)

    def run_parameter_sweep(self):
        count_valid = 0
        for i, X in enumerate(self.samples):
            if(i>=run_from):
            	ssq, store_me = self.evaluate_ssq(X,i,count_valid)
            	if(store_me):
                	count_valid = count_valid + 1
    
    def evaluate_ssq(self, parameter_values,attempt,count_valid):
        self.simulation.clearResults()
        self.simulation.resetParameters()
        for i, k in enumerate(self.constant_parameter_names):
            self.constants[k] = 10.0**parameter_values[i]
        try:
            self.simulation.run()
        except:
            print("Run error, skipping", attempt)
            store_me = False
            ssq = np.zeros(num_series+1)
            return ssq, store_me
        end_time_solution = np.zeros(num_series)
        valid_solution = np.full(num_series,False, dtype = bool)
        ssq = np.zeros(num_series+1)
        for i in range(0,num_series):
            end_time_solution[i] = self.simulation.results().states()[expt_state_uri[i]].values()[-1] #Last entry for each simulation result
            #Is this solution within a standard deviation of the mean?
            lower_bound =exp_data[i,0] - exp_data[i,1]
            upper_bound = exp_data[i,0] + exp_data[i,1]
            if(lower_bound <= end_time_solution[i] <= upper_bound):
                valid_solution[i] = True
                ssq[i+1] = (end_time_solution[i]-exp_data[i,0])**2.
        if(valid_solution.all()):
            ssq[0] = np.sum(ssq[1:(num_series+1)])
            store_me = True
            print('found a solution at attempt', attempt)
            print('ssq for this solution is', ssq[0])
            export_parameters= export_directory + export_parameter_filename + str(count_valid) + '.txt'
            export_solution= export_directory + export_solution_filename + str(count_valid) + '.txt'
            self.store_solution(self.simulation.results().states(),parameter_values,export_parameters,export_solution,ssq,end_time_solution)
        else:
            store_me = False
            print('Not storing, attempt: ',attempt)
            #You could put a print statement here to track progress
        
        return ssq, store_me
    
    def store_solution(self, results, parameter_values,export_parameters,export_solution,ssq,end_time_solution):
        f = open(export_parameters, 'a')
        for j, k in enumerate(self.constant_parameter_names):
            f.write('  {}: {:g} '.format(k, 10.0**parameter_values[j]))
            f.write('\n')
        f.write('\n')
        f.write('Ssq' + str(ssq))
        f.close()
        f1 = open(export_solution, 'a')
        for i in range(0,num_series):
            time_save = times[0::time_export]
            results_save = results[expt_state_uri[i]].values()[0::time_export]
            ss_val = len(results_save)
            result_diff = 1000.
            for j in reversed(results_save):
                ss_val = ss_val-1
                result_diff = (end_time_solution[i]-results_save[ss_val])/end_time_solution[i]
                if(abs(result_diff)>=1e-3):
                    break
            f1.write('Cytokine: ' + str(i) + '\n')
            f1.write('Time: ' + str(time_save) + '\n')
            f1.write('Solution : ' + str(results_save) + '\n')
            f1.write('Steady state at:' + str(time_save[ss_val]) + '\n')
            f1.write('ssq:' + str(ssq[i+1]) + '\n')
        f1.close()
            

                    
            
                   

s = Simulation()
v = s.run_parameter_sweep()

