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
        if np.random.rand() > 0.5:
            length_label = 'a'
        else:
            length_label = 'b'
        if len(all_recorded) - start_index > 2:
            next_length = Length.create_from_recorded(all_recorded, start_index + 1)
        else:
            next_length = None
        
        return Length(all_recorded[start_index], all_recorded[start_index + 1], next_length, True, True, length_label, all_recorded, start_index, start_index)
              
    #This function is expensive, run only once at start, then each length modifies global state in place as required
    def calculate_global_state(self, global_state={'missed': 0, 'recorded': 0, 'a': 0,
                                                   'b': 0, 'extra': 0, 'a_mean': 0.0, 'a_meansquare': 0.0,
                                                   'b_mean': 0.0, 'b_meansquare': 0.0}):
        
        global_state['missed'] += (not self.start_recorded)
        if self.next_length is None:
            global_state['missed'] += (not self.end_recorded)
        global_state['recorded'] += self.start_recorded
        if self.next_length is None:
            global_state['recorded'] += self.end_recorded
        global_state[self.length_label] += 1
        global_state['extra'] += (self.index_stop - self.index_start)
        
        global_state[self.length_label+'_mean'] = (global_state[self.length_label+'_mean']*(global_state[self.length_label]-1) + self.duration) / global_state[self.length_label]
        global_state[self.length_label+'_meansquare'] += (global_state[self.length_label+'_meansquare']*(global_state[self.length_label]-1) + self.duration**2) / global_state[self.length_label]
        
        self.global_state = global_state
        if not (self.next_length is None):
            self.next_length.calculate_global_state(global_state)