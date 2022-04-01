# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:34:56 2022

@author: tobycrisford
"""

import numpy as np

class Length:
    
    def __init__(self, start_time, finish_time, next_length, start_recorded, end_recorded, length_label, all_recorded, index_start, index_stop):
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
    
    @classmethod
    def create_from_recorded(cls, all_recorded, start_index):
        length_label = int(np.random.rand() > 0.5)
        if len(all_recorded) - start_index > 2:
            next_length = Length.create_from_recorded(all_recorded, start_index + 1)
        else:
            next_length = None
        
        return Length(all_recorded[start_index], all_recorded[start_index + 1], next_length, True, True, length_label, all_recorded, start_index, start_index)
              
    #This function is expensive, run only once at start, then each length modifies global state in place as required
    def calculate_global_state(self, global_state={'missed': 0, 'recorded': 0, 'n': np.array([0,0]), 'extra': 0, 'mean': np.array([0.0,0.0]), 'meansquare': np.array([0.0,0.0])}):
        
        global_state['missed'] += (not self.start_recorded)
        if self.next_length is None:
            global_state['missed'] += (not self.end_recorded)
        if not (self.next_length is None):
            global_state['recorded'] += self.end_recorded
        global_state['n'][self.length_label] += 1
        global_state['extra'] += (self.index_stop - self.index_start) + 1 - self.start_recorded
        
        global_state['mean'][self.length_label] = (global_state['mean'][self.length_label]*(global_state['n'][self.length_label]-1) + self.duration) / global_state['n'][self.length_label]
        global_state['meansquare'][self.length_label] = (global_state['meansquare'][self.length_label]*(global_state['n'][self.length_label]-1) + self.duration**2) / global_state['n'][self.length_label]
        global_state['S'] = global_state['n'] * (global_state['meansquare'] - global_state['mean']**2)
        
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
        potential_S = potential_n * (potential_meansquare - potential_mean**2)
                
        #prob_factor *= potential_S**(0.5*(1-potential_n)) / (self.global_state['S']**(0.5*(1-self.global_state['n'])))
        s_prob_factors = (potential_S/self.global_state['S'])**(0.5*(1-self.global_state['n']))
        s_prob_factors *= potential_S**(-0.5*(potential_n - self.global_state['n']))
        
        return s_prob_factors, potential_mean, potential_meansquare, potential_S
    
    def add_missing(self, potential_missed, new_start_index, new_stop_index):
        prob_factor = (self.global_state['missed'] + 1) / (self.global_state['missed'] + self.global_state['recorded'] + 2)
        prob_factor = prob_factor * ((self.global_state['n'] + 1) / (np.sum(self.global_state['n']) + 2))
                
        s_prob_factors, potential_mean, potential_meansquare, potential_S = self.s_effect_of_new_add(potential_missed)

        prob_factor *= s_prob_factors.prod(axis=1)
                
        prob_factor *= (1-(1/(2.0*self.global_state['n'])))*np.sqrt(1/(1+(1/self.global_state['n'])))
                
        prob_stick = 1 / (1 + np.sum(prob_factor))
        rand_number = np.random.rand()
        if rand_number > prob_stick:
            if rand_number < 1 - (prob_stick * prob_factor[1]):
                new_label = 0
            else:
                new_label = 1
            self.next_length = Length(potential_missed, self.finish_time,
                                      self.next_length, False, self.end_recorded,
                                      new_label, self.all_recorded, new_start_index, self.index_stop)
            self.next_length.global_state = self.global_state
            self.finish_time = potential_missed
            self.duration = self.finish_time - self.start_time
            self.end_recorded = False
            self.index_stop = new_stop_index #If < self.index_start this indicates no recorded points in [)
                    
            #Finally need to increment global state
            self.global_state['missed'] += 1
            self.global_state['n'][new_label] += 1
            self.global_state['mean'] = potential_mean[new_label,:]
            self.global_state['meansquare'] = potential_meansquare[new_label,:]
            self.global_state['S'] = potential_S[new_label,:]
                    
            return self.next_length
        
        return None
    
    def random_hop(self):
        
        #Should we change the current length label?
        prob_factor = (1+self.global_state['n'][1-self.length_label])/(self.global_state['n'][self.length_label])
        potential_mean = np.copy(self.global_state['mean'])
        potential_meansquare = np.copy(self.global_state['meansquare'])
        n_factor = self.global_state['n'][self.length_label] / (self.global_state['n'][self.length_label] - 1)
        potential_mean[self.length_label] -= self.duration / self.global_state['n'][self.length_label]
        potential_mean[self.length_label] *= n_factor
        potential_meansquare[self.length_label] -= (self.duration**2) / self.global_state['n'][self.length_label]
        potential_meansquare[self.length_label] *= n_factor
        potential_mean[1-self.length_label] = ((potential_mean[1-self.length_label] * self.global_state['n'][1-self.length_label]) + (self.duration)) / (self.global_state['n'][1-self.length_label] + 1)
        potential_meansquare[1-self.length_label] = ((potential_meansquare[1-self.length_label] * self.global_state['n'][1-self.length_label]) + (self.duration)**2) / (self.global_state['n'][1-self.length_label] + 1)
        potential_n = np.copy(self.global_state['n'])
        potential_n[self.length_label] -= 1
        potential_n[1-self.length_label] += 1
        potential_S = potential_n * (potential_meansquare - potential_mean**2)
                
        s_prob_factors = (potential_S/self.global_state['S'])**(0.5*(1-self.global_state['n']))
        s_prob_factors *= potential_S**(-0.5*(potential_n - self.global_state['n']))
        
        prob_factor *= np.prod(s_prob_factors)
        
        prob_factor *= (1-(1/(2.0*self.global_state['n'][1-self.length_label])))*np.sqrt(1/(1+(1/self.global_state['n'][1-self.length_label])))
        prob_factor *= ((self.global_state['n'][self.length_label] - 1) / (self.global_state['n'][self.length_label] - (3/2))) * np.sqrt(self.global_state['n'][self.length_label] / (self.global_state['n'][self.length_label] - 1))
        
        prob_stick = 1 / (1 + prob_factor)
        rand_number = np.random.rand()
        if rand_number > prob_stick:
            self.length_label = 1 - self.length_label
            self.global_state['n'] = potential_n
            self.global_state['mean'] = potential_mean
            self.global_state['meansquare'] = potential_meansquare
            self.global_state['S'] = potential_S
            return self
        
        
        for i in range(self.index_start, self.index_stop + 1):
            
            #Was a length transition missed?
            if i == self.index_start:
                lb = self.start_time
            else:
                lb = self.all_recorded[i-1]
            if self.all_recorded[i] > lb:
                potential_missed = np.random.uniform(lb, self.all_recorded[i])
                add_missing_test = self.add_missing(potential_missed, i, i - 1)
                if not (add_missing_test is None):
                    return add_missing_test
                
                
                #Should we add back in an extra recorded?
                prob_factor = (self.global_state['recorded'] + 1) / (self.global_state['missed'] + self.global_state['recorded'] + 2)
                if self.global_state['extra'] == 1:
                    prob_factor *= 2
                else:
                    prob_factor *= self.global_state['extra'] / (self.global_state['extra'] - 1)
                prob_factor = prob_factor * ((self.global_state['n'] + 1) / (np.sum(self.global_state['n']) + 2))
                
                s_prob_factors, potential_mean, potential_meansquare, potential_S = self.s_effect_of_new_add(self.all_recorded[i])
                
                prob_factor *= s_prob_factors.prod(axis=1)
                
                prob_factor *= (1-(1/(2.0*self.global_state['n'])))*np.sqrt(1/(1+(1/self.global_state['n'])))
                
                prob_stick = 1 / (1 + np.sum(prob_factor))
                rand_number = np.random.rand()
                if rand_number > prob_stick:
                    if rand_number < 1 - (prob_stick * prob_factor[1]):
                        new_label = 0
                    else:
                        new_label = 1
                    self.next_length = Length(self.all_recorded[i], self.finish_time,
                                              self.next_length, True, self.end_recorded,
                                              new_label, self.all_recorded, i, self.index_stop)
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
                    
                    return self.next_length
                
                
        #Was a length transition missed near the end?
        if self.index_stop < self.index_start:
            potential_missed = np.random.uniform(self.start_time, self.finish_time)
        else:
            potential_missed = np.random.uniform(self.all_recorded[self.index_stop], self.finish_time)
        add_missing_test = self.add_missing(potential_missed, self.index_stop + 1, self.index_stop)
        if not (add_missing_test is None):
            return add_missing_test
        
        #Should we merge with the next length?
        if not (self.next_length is None):
            if self.end_recorded:
                prob_factor = (self.global_state['missed']+self.global_state['recorded']+1)/(self.global_state['recorded'])
                if self.global_state['extra'] == 0:
                    prob_factor *= 0.5
                else:
                    prob_factor *= self.global_state['extra'] / (self.global_state['extra'] + 1)
            else:
                prob_factor = (self.global_state['missed']+self.global_state['recorded']+1)/(self.global_state['missed'])
            
            prob_factor *= (np.sum(self.global_state['n']) + 1) / self.global_state['n'][self.next_length.length_label]
            
            potential_mean = np.copy(self.global_state['mean'])
            potential_meansquare = np.copy(self.global_state['meansquare'])
            
            potential_mean[self.length_label] +=  (self.next_length.duration) / (self.global_state['n'][self.length_label])
            potential_meansquare[self.length_label] += (self.next_length.duration*(self.next_length.duration + 2*self.duration)) / (self.global_state['n'][self.length_label])
            
            n_factor = self.global_state['n'][self.next_length.length_label] / (self.global_state['n'][self.next_length.length_label] - 1)
            potential_mean[self.next_length.length_label] -= self.next_length.duration / self.global_state['n'][self.next_length.length_label]
            potential_mean[self.next_length.length_label] *= n_factor
            potential_meansquare[self.next_length.length_label] -= (self.next_length.duration**2) / self.global_state['n'][self.next_length.length_label]
            potential_meansquare[self.next_length.length_label] *= n_factor

            potential_n = np.copy(self.global_state['n'])
            potential_n[self.next_length.length_label] -= 1
            
            potential_S = potential_n * (potential_meansquare - potential_mean**2)
                    
            s_prob_factors = (potential_S/self.global_state['S'])**(0.5*(1-self.global_state['n']))
            s_prob_factors *= potential_S**(-0.5*(potential_n - self.global_state['n']))
            
            prob_factor *= np.prod(s_prob_factors)
            
            prob_factor *= ((self.global_state['n'][self.next_length.length_label] - 1) / (self.global_state['n'][self.next_length.length_label] - (3/2))) * np.sqrt(self.global_state['n'][self.next_length.length_label] / (self.global_state['n'][self.next_length.length_label] - 1))
            
            prob_stick = 1 / (1 + np.sum(prob_factor))
            rand_number = np.random.rand()
            if rand_number > prob_stick:
                if self.end_recorded:
                    self.global_state['recorded'] -= 1
                    self.global_state['extra'] += 1
                self.global_state['n'][self.next_length.length_label] -= 1
                self.global_state['mean'] = potential_mean
                self.global_state['meansquare'] = potential_meansquare
                self.global_state['S'] = potential_S
                
                self.finish_time = self.next_length.finish_time
                self.duration = self.finish_time - self.start_time
                self.end_recorded = self.next_length.end_recorded
                if self.index_stop < self.index_start:
                    self.index_start = self.next_length.index_start
                    self.index_stop = self.next_length.index_stop
                else:
                    self.index_stop = self.next_length.index_stop
                self.next_length = self.next_length.next_length
                
                return self
            
        return self.next_length
    
    
    #For debugging
    def print_all(self):
        
        print(self.start_time, self.finish_time)
        if not (self.next_length is None):
            self.next_length.print_all()
                