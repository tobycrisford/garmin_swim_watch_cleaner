# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:34:56 2022

@author: tobycrisford
"""

import numpy as np

class length:
    
    def __init__(self, start_time, finish_time, next_length, start_recorded, end_recorded, start_label, end_label):
        self.start_time = start_time
        self.finish_time = finish_time
        self.duration = finish_time - start_time
        self.next_length = next_length
        self.start_recorded = start_recorded
        self.end_recorded = end_recorded
        self.start_label = start_label
        self.end_label = end_label
    
    @classmethod
    def create_from_recorded(cls, recorded_times, start_label):
        end_label = int(np.random.rand() > 0.5)
        if len(recorded_times) > 2:
            next_length = length.create_from_recorded(recorded_times[1:], end_label)
        else:
            next_length = None
        
        return length(recorded_times[0], recorded_times[1], next_length, True, True, start_label, end_label)
        
