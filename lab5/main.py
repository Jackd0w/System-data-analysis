import numpy as np

import time

import numpy as np
import random
import math
from joblib import Parallel, delayed



class Bees:
    
    def __init__(self, bees, width = None):
        #, tolerable_go_out_of_bounds = False
        #if tolerable_go_out_of_bounds:
        #    self.update = lambda x,v: x+v
        #else:
        #    def upd(x, v):
        #        new = x+v




        if width == None:
            #width = np.absolute(bees).max()
            width = (bees.max()-bees.min())*100/bees.shape[0]
            
        
        self.x = np.array(bees)
        self.v = (np.random.random(self.x.shape) - 0.5) * width
        self.bests = self.x.copy()
        
    @staticmethod
    def get_Bees_from_randomputs(count, random_gen, width = None):
        bees = np.array([random_gen() for _ in range(count)])
        res = Bees(bees, width)
        return res
        
        
    def set_function(self, f, parallel = False):
        self.f = f
        
        self.get_vals = (lambda: np.array(Parallel(n_jobs=-1)(delayed(f)(v) for v in self.x))) if parallel else (lambda: np.array([f(v) for v in self.x]))
        
        self.vals = self.get_vals()
        
    def make_step(self, w, fp, fg, best_pos, best_val):
           
        self.x += self.v
        
        
        new_vals = self.get_vals()
        inds = new_vals < self.vals
        self.bests[inds,:] = self.x[inds,:].copy()
        
        minimum = new_vals.min()
        if minimum < best_val:
            best_val = minimum
            best_pos = self.bests[new_vals.argmin(),:].flatten().copy()
            #print(best_pos)
        self.vals[inds] = new_vals[inds]
        
        
        
        fi = fg + fp
        coef = 2*w/math.fabs(2-fi-math.sqrt(fi*(fi-4)))
        
        self.v = coef * (self.v + 
                         fp * np.random.random(self.x.shape)*(self.bests - self.x) + 
                         fg * np.random.random(self.x.shape)*(best_pos - self.x) )
        
        return best_val, best_pos
    
    def show(self):
        print("Current bees' positions:")
        print(self.x)
        print()
        print("Current bees' speeds:")
        print(self.v)
        print()
        
        if hasattr(self, 'vals'):
            print("Best bees' positions and values:")
            for p, v in zip(self.bests, self.vals):
                print(f'{p} --> {v}')
            print()
        



class Hive:
    
    def __init__(self, bees, func,  parallel = False,  verbose = True):
        
        self.bees = bees
        self.bees.set_function(func, parallel)
        
        
        
        self.best_pos = bees.bests[bees.vals.argmin(),:].flatten().copy()
        self.best_val = bees.vals.min()
        
        
        if verbose:
            print(f"total bees: {self.bees.x.shape[0]}")
            print(f"best value (at beggining): {self.best_val}")
        
    def get_result(self, max_step_count = 100, max_fall_count = 30, w = 0.3, fp = 2, fg = 5, latency = 1e-9, verbose = True, max_time_seconds = None):
        
        start_time = time.time()
        times_done = (lambda: False) if max_time_seconds is None else (lambda: time.time() - start_time > max_time_seconds)
        
        if latency != None:
            latency = 1 - latency
        
        w = 1 if math.fabs(w)>1 else math.fabs(w)
        
        sm = 4/(fp+fg)
        if sm > 1:
            fp, fg = fp*sm, fg*sm
        
        
        
        count_fall = 0
        val = self.best_val
        
        for i in range(1,max_step_count+1):
            self.best_val, self.best_pos = self.bees.make_step(w, fp, fg, self.best_pos, self.best_val)
            
            if self.best_val < val:     
                
                if latency != None and self.best_val/val>latency:
                    #print(f'{self.best_val/val} > {latency}')
                    #if verbose:
                    #    print(f'I should stop if new_val/old_val > {max_new_percent} (now {self.best_val/val})')
                    #return self.best_val, self.best_pos
                    count_fall+=1
                
                if verbose:
                    print(f'new best value = {self.best_val} after {i} iteration')
                    val = self.best_val
                

            else:
                
                count_fall+=1
                
            if count_fall == max_fall_count:
                    
                if verbose:
                    print(f'I should stop after {count_fall} fallen iterations')
                    
                return self.best_val, self.best_pos
            
            if times_done():
                print('Time is done!')
                return self.best_val, self.best_pos
        
        return self.best_val, self.best_pos
    
    def show(self):
        self.bees.show()
        print(f'Best value: {self.best_val}')
        print(f'Best position: {self.best_pos}')
        
            

class BeeHive:
    @staticmethod
    def Minimize(func, bees, 
                 max_step_count = 100, max_fall_count = 30, 
                 w = 0.3, fp = 2, fg = 5, latency = 1e-9, 
                 verbose = True, parallel = False, max_time_seconds = None):
        #bees = Bees(bees, width)
        hive = Hive(bees, func, parallel , verbose)
        
        return hive.get_result(max_step_count, max_fall_count, w, fp,fg, latency, verbose, max_time_seconds=max_time_seconds)

class TestFunctions:
    @staticmethod
    def Parabol(arr):
        return np.sum(arr**2)
    @staticmethod
    def Rastrigin(arr):
        return 10*arr.size+TestFunctions.Parabol(arr) - 10*np.sum(np.cos(2*math.pi*arr))
    @staticmethod
    def Shvel(arr):
        return -np.sum(arr*np.sin(np.sqrt(np.abs(arr))))

class RandomPuts:
    @staticmethod
    def Uniform(minimum, maximum, size):
        return lambda: np.random.uniform(minimum, maximum, size)
    @staticmethod
    def Normal(mean, std, size):
        return lambda: np.random.normal(mean, std, size)

f1 = lambda arr: arr[0]+arr[1]/(1+arr[0])+arr[2]*arr[3]

# convertion to numpy->float function

def target(x,y,z,q):
  return x**2+y**2*z/q

f2 = lambda arr: target(arr[0], arr[1], arr[2], arr[3])

f_tmp = lambda arr: -target(arr)

#tagret_result = -global_min

np.random.seed(1)

# it's just numpy array with shape bees_count_x_dim

arr = np.random.uniform(low = -3, high = 5, size = (10,3))

# width parameter means the maximum range of random begging speeds

bees = Bees(arr, width = 0.2)

bees.show()

bees = Bees(np.random.normal(loc = 2, scale = 2, size = (100,3)), width = 3)

func = lambda arr: TestFunctions.Parabol(arr-3)

    
hive = Hive(bees, 
            func, 
            parallel = False,  
            verbose = True)  

best_result, best_position = hive.get_result(max_step_count = 25, # maximun count of iteraions
                      max_fall_count = 6, # maximum count of continious iterations without better result
                      w = 0.3, fp = 2, fg = 5, # parameters of algorithm
                      latency = 1e-9, # if new_result/old_result > 1-latency then it was the iteration without better result
                      verbose = True, # show the progress
                      max_time_seconds = None # max seconds of working 
                      )





# u also can use this code (without creating a hive)

best_result, best_position = BeeHive.Minimize(func, bees, 
                 max_step_count = 100, max_fall_count = 30, 
                 w = 0.3, fp = 2, fg = 5, latency = 1e-9, 
                 verbose = False, parallel = False)

bees = Bees.get_Bees_from_randomputs(count = 10, random_gen = RandomPuts.Normal(mean = 1, std = 0.1, size = 2), width = 0.3)

bees.show()

func = lambda arr: TestFunctions.Rastrigin(arr) + TestFunctions.Shvel(arr) + 1 / (1 + np.sum(np.abs(arr)))

for w in (0.1,0.3,0.5,0.8):
    for fp in (1, 2, 3, 4.5):
        for fg in (3, 5, 8, 15):
            
            # 200 bees, 10 dimentions
            bees = Bees(np.random.uniform(low = -100, high = 100, size = (200, 10) ))
            
            best_val, _ = BeeHive.Minimize(func, bees, 
                 max_step_count = 200, max_fall_count = 70, 
                 w = 2, fp = fp, fg = fg, latency = 1e-9, 
                 verbose = False, parallel = False)
            
            print(f'best val by w = {w}, fp = {fp}, fg = {fg} is {best_val}')

if __name__ == '__main__':
    
    bs = (np.random.random((200,10))-0.5)*15
    bees = Bees(bs)
    
    #bees.show()
    
    #bees = Bees.get_Bees_from_randomputs(200, RandomPuts.Uniform(-3,10, size = 10))
    
    #bees = Bees.get_Bees_from_randomputs(200, RandomPuts.Normal(3,2, size = 10))
    
    func = lambda arr: TestFunctions.Parabol(arr-3)
    
    hive = Hive(bees,func, parallel = False , verbose = True)
    
    res = hive.get_result(500)