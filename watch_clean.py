# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:34:56 2022

@author: tobycrisford
"""

import numpy as np
from tqdm import tqdm
import copy

class Length:
    
    def __init__(self, start_time, finish_time, next_length, start_recorded, end_recorded, length_label, all_recorded, index_start, index_stop, start_moved):
        #index_start/stop gives the lowest, highest index of all_recorded in [...) (if stop < start this indicates no values)
        self.start_time = start_time
        self.finish_time = finish_time
        self.duration = finish_time - start_time
        self.next_length = next_length
        self.start_recorded = start_recorded
        self.end_recorded = end_recorded
        self.length_label = length_label
        self.all_recorded = all_recorded
        self.index_start = index_start
        self.index_stop = index_stop
        self.start_moved = start_moved
    
    @classmethod
    def create_from_recorded(cls, all_recorded, start_index=0):
        length_label = int(np.random.rand() > 0.5)
        if len(all_recorded) - start_index > 2:
            next_length = Length.create_from_recorded(all_recorded, start_index + 1)
        else:
            next_length = None
        
        return Length(all_recorded[start_index], all_recorded[start_index + 1], next_length, True, True, length_label, all_recorded, start_index, start_index, False)
              
    #This function is expensive, run only once at start, then each length modifies global state in place as required
    def calculate_global_state(self, global_state=None):
        
        if global_state is None:
            global_state = {'missed': 0, 'recorded': 0, 'n': np.array([0,0]), 'extra': 0, 'mean': np.array([0.0,0.0]), 'meansquare': np.array([0.0,0.0]), 'moved_starts': 0}
        
        if not self.start_moved:
            global_state['missed'] += (not self.start_recorded)
        else:
            global_state['moved_starts'] += 1
        if self.next_length is None:
            global_state['missed'] += (not self.end_recorded)
        if not (self.next_length is None):
            global_state['recorded'] += self.end_recorded
        global_state['n'][self.length_label] += 1
        if self.index_stop >= self.index_start:
            global_state['extra'] += (self.index_stop - self.index_start) + 1 - self.start_recorded
        
        global_state['mean'][self.length_label] = (global_state['mean'][self.length_label]*(global_state['n'][self.length_label]-1) + self.duration) / global_state['n'][self.length_label]
        global_state['meansquare'][self.length_label] = (global_state['meansquare'][self.length_label]*(global_state['n'][self.length_label]-1) + self.duration**2) / global_state['n'][self.length_label]
        global_state['S'] = global_state['n'] * (global_state['meansquare'] - global_state['mean']**2)
        global_state['S'][global_state['S'] <= 1] = 1 #regularize
        global_state['var'] = global_state['meansquare'] - global_state['mean']**2
        global_state['var'][global_state['var'] <= 10**(-5)] = 10**(-5)
        
        self.global_state = global_state
        if not (self.next_length is None):
            self.next_length.calculate_global_state(self.global_state)
            
    
    def s_effect_of_new_add(self, cut_time):
        
        potential_mean = np.array([self.global_state['mean'], self.global_state['mean']])
        potential_meansquare = np.array([self.global_state['meansquare'], self.global_state['meansquare']])
        potential_mean[:, self.length_label] -= (self.finish_time - cut_time) / self.global_state['n'][self.length_label]
        potential_meansquare[:, self.length_label] -= (self.duration**2 - (cut_time - self.start_time)**2) / self.global_state['n'][self.length_label]
        for ind in (0,1):
            potential_mean[ind,ind] = ((potential_mean[ind,ind] * self.global_state['n'][ind]) + (self.finish_time - cut_time)) / (self.global_state['n'][ind] + 1)
            potential_meansquare[ind,ind] = ((potential_meansquare[ind,ind] * self.global_state['n'][ind]) + (self.finish_time - cut_time)**2) / (self.global_state['n'][ind] + 1)
        potential_n = self.global_state['n'] + np.identity(2)
        potential_S, potential_var = self.compute_potential_S(potential_mean, potential_meansquare, potential_n)
                
        s_prob_factors = (potential_var/self.global_state['var'])**(0.5*(1-self.global_state['n']))
        s_prob_factors *= potential_var**(-0.5*(potential_n - self.global_state['n']))
        
        return s_prob_factors, potential_mean, potential_meansquare, potential_S, potential_var
    
    def add_missing(self, potential_missed, new_start_index, new_stop_index, time_factor, beta, near_extra):
        if not near_extra:
            prob_factor = (self.global_state['missed'] + 1) / (self.global_state['missed'] + self.global_state['recorded'] + (beta-1) + 2)
        prob_factor = prob_factor * ((self.global_state['n'] + 1) / (np.sum(self.global_state['n']) + 2))
                
        s_prob_factors, potential_mean, potential_meansquare, potential_S, potential_var = self.s_effect_of_new_add(potential_missed)

        prob_factor *= s_prob_factors.prod(axis=1)
                
        regularized_n = np.copy(self.global_state['n'])
        regularized_n[regularized_n == 0] = 1
        prob_factor *= (regularized_n / (1 + regularized_n)) * (1 / np.sqrt(2*np.pi*np.e))
        prob_factor *= time_factor
                
        prob_stick = 1 / (1 + np.sum(prob_factor))
        rand_number = np.random.rand()
        if rand_number > prob_stick:
            if rand_number < 1 - (prob_stick * prob_factor[1]):
                new_label = 0
            else:
                new_label = 1
            self.next_length = Length(potential_missed, self.finish_time,
                                      self.next_length, False, self.end_recorded,
                                      new_label, self.all_recorded, new_start_index, self.index_stop, True)
            self.next_length.global_state = self.global_state
            self.finish_time = potential_missed
            self.duration = self.finish_time - self.start_time
            self.end_recorded = False
            self.index_stop = new_stop_index #If < self.index_start this indicates no recorded points in [)
                    
            #Finally need to increment global state
            if near_extra:
                self.global_state['moved_starts'] += 1
            else:
                self.global_state['missed'] += 1
            self.global_state['n'][new_label] += 1
            self.global_state['mean'] = potential_mean[new_label,:]
            self.global_state['meansquare'] = potential_meansquare[new_label,:]
            self.global_state['S'] = potential_S[new_label,:]
            self.global_state['var'] = potential_var[new_label,:]
                    
            return self.next_length
        
        return None
    
    def compute_potential_S(self, potential_mean, potential_meansquare, potential_n):
        
        potential_S = potential_n * (potential_meansquare - potential_mean**2)
        
        potential_S[potential_S <= 1] = 1 #To regularize
        
        potential_var = potential_meansquare - potential_mean**2
        
        potential_var[potential_var <= 10**-5] = 10**(-5) #To regularize
        
        return potential_S, potential_var
        
    
    def random_hop(self, beta, watch_min):
        
        #Should we change the current length label?
        prob_factor = (1+self.global_state['n'][1-self.length_label])/(self.global_state['n'][self.length_label])
        potential_mean = np.copy(self.global_state['mean'])
        potential_meansquare = np.copy(self.global_state['meansquare'])
        if self.global_state['n'][self.length_label] > 1:
            n_factor = self.global_state['n'][self.length_label] / (self.global_state['n'][self.length_label] - 1)
        else:
            n_factor = 1
        potential_mean[self.length_label] -= self.duration / self.global_state['n'][self.length_label]
        potential_mean[self.length_label] *= n_factor
        potential_meansquare[self.length_label] -= (self.duration**2) / self.global_state['n'][self.length_label]
        potential_meansquare[self.length_label] *= n_factor
        potential_mean[1-self.length_label] = ((potential_mean[1-self.length_label] * self.global_state['n'][1-self.length_label]) + (self.duration)) / (self.global_state['n'][1-self.length_label] + 1)
        potential_meansquare[1-self.length_label] = ((potential_meansquare[1-self.length_label] * self.global_state['n'][1-self.length_label]) + (self.duration)**2) / (self.global_state['n'][1-self.length_label] + 1)
        potential_n = np.copy(self.global_state['n'])
        potential_n[self.length_label] -= 1
        potential_n[1-self.length_label] += 1
        potential_S, potential_var = self.compute_potential_S(potential_mean, potential_meansquare, potential_n)
                
        s_prob_factors = (potential_var/self.global_state['var'])**(0.5*(1-self.global_state['n']))
        s_prob_factors *= potential_var**(-0.5*(potential_n - self.global_state['n']))
        
        prob_factor *= np.prod(s_prob_factors)
        
        if self.global_state['n'][1-self.length_label] > 0:
            prob_factor *= (self.global_state['n'][1-self.length_label] / (1 + self.global_state['n'][1-self.length_label])) * (1 / np.sqrt(2*np.pi*np.e))
        if self.global_state['n'][self.length_label] > 1:
            prob_factor *= ((self.global_state['n'][self.length_label]) / (self.global_state['n'][self.length_label] - 1)) * np.sqrt(2*np.pi*np.e)
        
        prob_stick = 1 / (1 + prob_factor)
        rand_number = np.random.rand()
        if rand_number > prob_stick:
            self.length_label = 1 - self.length_label
            self.global_state['n'] = potential_n
            self.global_state['mean'] = potential_mean
            self.global_state['meansquare'] = potential_meansquare
            self.global_state['S'] = potential_S
            self.global_state['var'] = potential_var
            return self.next_length
        
        
        for i in range(self.index_start, self.index_stop + 1):
            
            #Was a length transition missed?
            if i == self.index_start:
                lb = self.start_time
                lb_extra = False
            else:
                lb = self.all_recorded[i-1]
                lb_extra = True
            if self.all_recorded[i] > lb:
                potential_missed = np.random.uniform(lb, self.all_recorded[i])
                add_missing_test = self.add_missing(potential_missed, i, i - 1, self.all_recorded[i] - lb, beta, (potential_missed - lb < watch_min) and lb_extra)
                if not (add_missing_test is None):
                    return add_missing_test
                
                
                #Should we add back in an extra recorded?
                prob_factor = (self.global_state['recorded'] + (beta-1) + 1) / (self.global_state['missed'] + self.global_state['recorded'] + (beta-1) + 2)
                if self.global_state['extra'] == 1:
                    prob_factor *= 2 #To regularize
                else:
                    prob_factor *= self.global_state['extra'] / (self.global_state['extra'] - 1)
                prob_factor = prob_factor * ((self.global_state['n'] + 1) / (np.sum(self.global_state['n']) + 2))
                
                s_prob_factors, potential_mean, potential_meansquare, potential_S, potential_var = self.s_effect_of_new_add(self.all_recorded[i])
                
                prob_factor *= s_prob_factors.prod(axis=1)
                
                regularized_n = np.copy(self.global_state['n'])
                regularized_n[regularized_n == 0] = 1
                prob_factor *= ((regularized_n) / (regularized_n+1)) * (1/np.sqrt(2*np.pi*np.e))
                
                prob_factor *= self.all_recorded[-1] - self.all_recorded[0]
                
                if i == self.index_stop and (not (self.next_length is None)):
                    if self.next_length.start_moved and self.next_length.start_time - self.all_recorded[i] < watch_min:
                        prob_factor *= (self.global_state['missed'] + 1) / (self.global_state['missed'] + (self.global_state['recorded'] + 1) + (beta-1) + 2)
                        
                
                prob_stick = 1 / (1 + np.sum(prob_factor))
                rand_number = np.random.rand()
                if rand_number > prob_stick:
                    if rand_number < 1 - (prob_stick * prob_factor[1]):
                        new_label = 0
                    else:
                        new_label = 1
                    if i == self.index_stop and (not (self.next_length is None)):
                        if self.next_length.start_moved and self.next_length.start_time - self.all_recorded[i] < watch_min:
                            self.next_length.start_moved = False
                            self.global_state['missed'] += 1
                            self.global_state['moved_starts'] -= 1
                    self.next_length = Length(self.all_recorded[i], self.finish_time,
                                              self.next_length, True, self.end_recorded,
                                              new_label, self.all_recorded, i, self.index_stop, False)
                    self.next_length.global_state = self.global_state
                    self.finish_time = self.all_recorded[i]
                    self.duration = self.finish_time - self.start_time
                    self.end_recorded = True
                    self.index_stop = i - 1
                    
                    self.global_state['recorded'] += 1
                    self.global_state['n'][new_label] += 1
                    self.global_state['extra'] -= 1
                    self.global_state['mean'] = potential_mean[new_label,:]
                    self.global_state['meansquare'] = potential_meansquare[new_label,:]
                    self.global_state['S'] = potential_S[new_label,:]
                    self.global_state['var'] = potential_var[new_label,:]
                    
                    return self.next_length
                
                
        #Was a length transition missed near the end?
        if self.index_stop < self.index_start:
            potential_missed = np.random.uniform(self.start_time, self.finish_time)
            add_missing_test = self.add_missing(potential_missed, self.index_stop + 1, self.index_stop, self.finish_time - self.start_time, beta, False)
        else:
            potential_missed = np.random.uniform(self.all_recorded[self.index_stop], self.finish_time)
            add_missing_test = self.add_missing(potential_missed, self.index_stop + 1, self.index_stop, self.finish_time - self.all_recorded[self.index_stop], beta,
                                                self.all_recorded[self.index_stop] != self.start_time and potential_missed - self.all_recorded[self.index_stop] < watch_min)
        if not (add_missing_test is None):
            return add_missing_test
        
        #Should we merge with the next length?
        if (not (self.next_length is None)):
            if self.end_recorded:
                prob_factor = (self.global_state['missed']+self.global_state['recorded']+1+(beta-1))/(self.global_state['recorded']+(beta-1))
                if self.global_state['extra'] == 0:
                    prob_factor *= 0.5
                else:
                    prob_factor *= self.global_state['extra'] / (self.global_state['extra'] + 1)
                prob_factor *= 1 / (self.all_recorded[-1] - self.all_recorded[0])
                if (not self.next_length.end_recorded) and self.next_length.duration < watch_min and self.next_length.index_start == self.next_length.index_stop:
                    prob_factor *= (self.global_state['missed'] + (self.global_state['recorded']-1) + 1 + (beta-1)) / (self.global_state['missed'])
            else:
                if self.next_length.start_moved:
                    prob_factor = 1.0
                else:
                    prob_factor = (self.global_state['missed']+self.global_state['recorded']+1+(beta-1))/(self.global_state['missed'])
                if self.next_length.index_stop < self.next_length.index_start:
                    ub = self.next_length.finish_time
                else:
                    ub = self.all_recorded[self.next_length.index_start]
                if self.index_stop < self.index_start:
                    lb = self.start_time
                else:
                    lb = self.all_recorded[self.index_stop]
                prob_factor *= 1 / (ub - lb)
            
            prob_factor *= (np.sum(self.global_state['n']) + 1) / self.global_state['n'][self.next_length.length_label]
            
            potential_mean = np.copy(self.global_state['mean'])
            potential_meansquare = np.copy(self.global_state['meansquare'])
            
            potential_mean[self.length_label] +=  (self.next_length.duration) / (self.global_state['n'][self.length_label])
            potential_meansquare[self.length_label] += (self.next_length.duration*(self.next_length.duration + 2*self.duration)) / (self.global_state['n'][self.length_label])
            
            if self.global_state['n'][self.next_length.length_label] > 1:
                n_factor = self.global_state['n'][self.next_length.length_label] / (self.global_state['n'][self.next_length.length_label] - 1)
            else:
                n_factor = 1
            potential_mean[self.next_length.length_label] -= self.next_length.duration / self.global_state['n'][self.next_length.length_label]
            potential_mean[self.next_length.length_label] *= n_factor
            potential_meansquare[self.next_length.length_label] -= (self.next_length.duration**2) / self.global_state['n'][self.next_length.length_label]
            potential_meansquare[self.next_length.length_label] *= n_factor

            potential_n = np.copy(self.global_state['n'])
            potential_n[self.next_length.length_label] -= 1
            
            potential_S, potential_var = self.compute_potential_S(potential_mean, potential_meansquare, potential_n)
                    
            s_prob_factors = (potential_var/self.global_state['var'])**(0.5*(1-self.global_state['n']))
            s_prob_factors *= potential_var**(-0.5*(potential_n - self.global_state['n']))
            
            prob_factor *= np.prod(s_prob_factors)
            
            if self.global_state['n'][self.next_length.length_label] > 1:
                prob_factor *= (self.global_state['n'][self.next_length.length_label] / (self.global_state['n'][self.next_length.length_label] - 1)) * np.sqrt(2*np.pi*np.e)
            
            prob_stick = 1 / (1 + np.sum(prob_factor))
            rand_number = np.random.rand()
            if rand_number > prob_stick:
                if self.end_recorded:
                    self.global_state['recorded'] -= 1
                    self.global_state['extra'] += 1
                    if (not self.next_length.end_recorded) and self.next_length.duration < watch_min and self.next_length.index_start == self.next_length.index_stop:
                        self.global_state['missed'] += 1
                        self.global_state['moved_starts'] -= 1
                        self.next_length.next_length.start_moved = False
                else:
                    if self.next_length.start_moved:
                        self.global_state['moved_starts'] -= 1
                    else:
                        self.global_state['missed'] -= 1
                self.global_state['n'][self.next_length.length_label] -= 1
                self.global_state['mean'] = potential_mean
                self.global_state['meansquare'] = potential_meansquare
                self.global_state['S'] = potential_S
                self.global_state['var'] = potential_var
                
                self.finish_time = self.next_length.finish_time
                self.duration = self.finish_time - self.start_time
                self.end_recorded = self.next_length.end_recorded
                if self.index_stop < self.index_start:
                    self.index_start = self.next_length.index_start
                    self.index_stop = self.next_length.index_stop
                else:
                    self.index_stop = self.next_length.index_stop
                self.next_length = self.next_length.next_length
                
                return self.next_length
            
        return self.next_length
    
    
    #For debugging
    def print_all(self):
        
        print(self.length_label, self.start_time, self.start_recorded, self.finish_time, self.end_recorded)
        for i in range(self.index_start, self.index_stop+1):
            if self.all_recorded[i] > self.start_time:
                print('Extra', self.all_recorded[i])
        if not (self.next_length is None):
            self.next_length.print_all()
            
    
    #For testing
    @classmethod
    def init_random_state(cls, n, a_length, b_length, extra_mean, miss_prob):
        t = [0]
        for i in range(1,n+1):
            if np.random.rand() > 0.5:
                t.append(t[i-1] + np.random.normal(a_length, 1.0))
            else:
                t.append(t[i-1] + np.random.normal(b_length, 1.0))
        
        t += list(np.random.uniform(t[0], t[-1], size = np.random.poisson(extra_mean)))
        
        for time in t:
            if np.random.rand() < miss_prob:
                t.remove(time)
                
        t.sort()
        
        return Length.create_from_recorded(t, 0)
    
    


    def run_increment(self, beta):
        inc = self.random_hop(beta)
        while not (inc is None):
            assert inc.global_state['missed'] + inc.global_state['recorded'] == np.sum(inc.global_state['n']) - 1
            inc = inc.random_hop(beta)

def run_monte_carlo(lengths, n_start, n_checkpoints, checkpoint_size, beta):
    
    for i in tqdm(range(n_start)):
        lengths.run_increment(beta)
        
    checkpoints = []
    
    for i in tqdm(range(n_checkpoints)):
        checkpoints.append(copy.deepcopy(lengths.global_state))
        for j in range(checkpoint_size):
            lengths.run_increment(beta)
            
    return checkpoints

def global_state_testing(lengths, n, beta):
    
    inc = lengths
    times_length = lengths.global_state['extra'] + lengths.global_state['recorded']
    
    for i in tqdm(range(n)):
        inc = inc.random_hop(beta)
        gs = copy.deepcopy(lengths.global_state)
        lengths.calculate_global_state()
        for j in gs:
            if isinstance(lengths.global_state[j], np.ndarray):
                if not np.all(np.isclose(gs[j], lengths.global_state[j])):
                    print(gs[j], lengths.global_state[j])
                    raise Exception("Global state inconsistency: " + j)
            elif gs[j] != lengths.global_state[j]:
                print(gs[j], lengths.global_state[j])
                raise Exception("Global state inconsistency: " + j)
        assert gs['missed'] + gs['recorded'] == np.sum(gs['n']) - 1
        assert gs['extra'] + gs['recorded'] == times_length
        if inc is None:
            inc = lengths
                