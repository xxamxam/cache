import numpy as np
from src.utils import CacheItem
from copy import deepcopy
class LRUCacheItem(CacheItem):
    """Data structure of items stored in cache"""
    def __init__(self, key, item):
        self.key = key
        self.item = item
   
class BeladyMin:
    def __init__(self, capacity, order, verbose = False):
        self.capacity = capacity
        self.cache = {}
        self.order = list(deepcopy(order))   # in these algorithm we now order if needed pages
        self.iteration = 0
        self.verbose = verbose
        if self.verbose:
            self.loads = 0
    def getItem(self, item):
        self.iteration += 1
        if item.key in self.cache:
            return item
        else: # find the most unnececary key as a victim
            if len(self.cache) > self.capacity:
                if self.verbose: self.loads += 1
                _key = None
                mmin = -np.inf
                for key in self.cache: 
                    try:
                        ind = self.order[self.iteration:].index(key)
                    except:
                        ind = np.inf  #not found
                    if ind > mmin:
                        mmin = ind
                        _key = key
                if _key is not None:
                    self.removeItem(_key)
            self.cache[item.key] = item
            return item
        
    def removeItem(self, key):
        del self.cache[key]
     
    def return_verbose(self):
        return {"loads": self.loads}
