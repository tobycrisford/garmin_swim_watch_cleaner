# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:34:56 2022

@author: tobycrisford
"""

import numpy as np

class Length:
    
    def __init__(self, start_time, finish_time, next_length, start_recorded, end_recorded, length_label, all_recorded, index_start, index_stop):
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
        global_state['extra'] += (self.index_stop - self.index_start)
        
        global_state['mean'][self.length_label] = (global_state['mean'][self.length_label]*(global_state['n'][self.length_label]-1) + self.duration) / global_state['n'][self.length_label]
        global_state['meansquare'][self.length_label] = (global_state['meansquare'][self.length_label]*(global_state['n'][self.length_label]-1) + self.duration**2) / global_state['n'][self.length_label]
        global_state['S'] = global_state['n'] * (global_state['meansquare'] - global_state['mean']**2)
        
        self.global_state = global_state
        if not (self.next_length is None):
            self.next_length.calculate_global_state(self.global_state)
            
    
    def random_hop(self):
        
        for i in range(self.index_start, self.index_stop):
            
            #Was a length transition missed?
            potential_missed = np.random.uniform(self.start_time, self.all_recorded[i])
            prob_factor = (self.global_state['missed'] + 1) / (self.global_state['missed'] + self.global_state['recorded'] + 2)
            prob_factor = prob_factor * ((self.global_state['n'] + 1) / (np.sum(self.global_state['n']) + 2))
            potential_mean = np.array([self.global_state['mean'], self.global_state['mean']])
            potential_meansquare = np.array(self.global_state['meansquare'], self.global_state['meansquare'])
            potential_mean[:, self.length_label] -= (self.finish_time - potential_missed) / self.global_state['n'][self.length_label]
            potential_meansquare[:, self.length_label] -= (self.duration**2 - (potential_missed - self.start_time)**2) / self.global_state['n'][self.length_label]
            for ind in (0,1):
                potential_mean[ind,ind] = ((potential_mean[ind,ind] * self.global_state['n'][ind]) + (self.finish_time - potential_missed)) / (self.global_state['n'][ind] + 1)
                potential_meansquare[ind,ind] = ((potential_meansquare[ind,ind] * self.global_state['n'][ind]) + (self.finish_time - potential_missed)**2) / (self.global_state['n'][ind] + 1)
            potential_S = potential_meansquare - potential_mean**2
            