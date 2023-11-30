from collections import OrderedDict
from copy import deepcopy
import numpy as np

class MyOrderedDict(OrderedDict):
    def __init__(self, max_size):
        super().__init__(self)
        self.max_size = max_size
    def __setitem__(self, key, value):
        while super().__len__() > self.max_size:
            super().popitem(last = False)
        super().__setitem__(key, deepcopy(value))
        self.move_to_end(key)
    def __getitem__(self, key):
        super().move_to_end(key)
        key, val = super().popitem()
        return val
    
class BanditMeta:
    def __init__(self):
        self.last_use = 0
        self.val = 0

class BanditCache:
    def __init__(self, capacity, history_capacity = None, eps1 = 0.1, eps2 = 0.1, verbose = False):
        self.capacity = capacity
        if history_capacity is None: history_capacity = capacity 
        
        self.cache = {}  #cache 
        self.size = 0

        self._cache_meta = {}  # key: BanditMeta
        self._history_meta = MyOrderedDict(history_capacity)

        self.eps1 = eps1
        self.eps2 = eps2

        self.sum = 0 # need for fast control
        self.iteration = 0
        self.verbose = verbose
        if verbose:
            self.loads = 0
            self.reuse_rate = 0

    def return_verbose(self):
        return {"loads": self.loads, 'reuse': self.reuse_rate}
    
    def __remove_from_cache(self, _key):
        try:
            del self.cache[_key]
            self.size -=1
            self.sum -= self._cache_meta[_key].val
            del self._cache_meta[_key]
        except KeyError:
            raise KeyError() from None
    def __insert_to_cache(self, _key, _value, cache_meta):
        try:
            self.cache[_key] = _value
            self.size += 1
            self.sum += cache_meta.val
            self._cache_meta[_key] = cache_meta
        except KeyError:
            raise KeyError() from None 
    def __update_in_cache(self, _key, _value, cache_meta_item):
        try:
            if _key not in self.cache: raise ValueError()
            self.cache[_key] = _value
            self.sum += cache_meta_item.val - self._cache_meta[_key].val  # update information
            self._cache_meta[_key] = cache_meta_item
        except KeyError:
            raise KeyError() from None 
    def __update_meta_item(self, item, eps):
        item = deepcopy(item)
        val =( self.sum/self.size )* np.exp(-(np.log(self.iteration /item.last_use))*eps / (self.sum/self.size))
        item.val = val
        item.last_use = self.iteration
        return item
        
    def __remove_victim(self):
        try:
            if self.verbose: self.loads += 1

            # p = np.array([it.val for it in self._cache_meta.values()])
            # p = p/np.sum(p)
            # victim_key = np.random.choice(list(self._cache_meta.keys()), 1, p = p)[0] # 

            victim_key = max(self._cache_meta, key = lambda x: self._cache_meta[x].val)
            self._history_meta[victim_key] = deepcopy(self._cache_meta[victim_key])
            self.__remove_from_cache(victim_key)
        except KeyError:
            raise KeyError() from None 
        
    def __check_sum(self):
        if self.sum > 1e3 or self.sum <= 1e-3:
            tmp = 0.
            for _key in self._cache_meta:
                self._cache_meta[_key].val /= self.sum
                tmp += self._cache_meta[_key].val
            self.sum = tmp
        return
    
    def __setitem__(self, _key, _value):
        self.iteration += 1
        if _key in self.cache:
            new_item = self.__update_meta_item(self._cache_meta[_key], self.eps2)
            self.__update_in_cache(_key, _value, new_item)
        else:
            while self.size >= self.capacity:
                self.__remove_victim() # select victim and delete it from cache and cache meta
            item = BanditMeta()
            if _key in self._history_meta:
                if self.verbose: self.reuse_rate += 1
                item = self.__update_meta_item(self._history_meta[_key], self.eps1)
            else:
                item.last_use = self.iteration
                if self.size == 0:
                    item.val = 1.
                else:
                    item.val = self.sum/self.size
            self.__insert_to_cache(_key, _value, item)
        self.__check_sum()
        return 
            
        
