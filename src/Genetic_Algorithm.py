from State import State
import numpy as np

class Score:
    def __init__(self, s_S_List, score):
        self.vector = s_S_List
        self.score = score
    
    def compare(self, other):
        """Compares score of current and other, returns true if self > other

        Args:
            other (Score): other score

        Returns:
            Boolean: True or False
        """
        return self.score > other.score
    

class Genetic_Algoritgm:
    def __init__(self, state):
        self.state = state
    
    def evaluate_fitness(self, arr, iterations):
        """Finds the average score of i iterations

        Args:
            arr (list): s_S List
            iterations (int): Number of iterations

        Returns:
            int: average score over i iterations
        """
        score = 0
        self.state.take_vector(arr)
        for i in range(iterations):
            self.state.reset()
            self.state.run()
            score += self.state.total_sum()
        return score / iterations
    
    
    
        
        
        
    