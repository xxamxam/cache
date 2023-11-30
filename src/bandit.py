import numpy as np
from utils import CacheItem


class CircularList(object):
    def __init__(self, len):
        self.len = len
        self.pos = 0
        self.data = [None for i in range(len)]
    def __getitem__(self, index):
        index = index % self.len
        return self.data[index]
    def setitem(self, item): # меняет элемент с самым старым
        tmp = self.data[self.pos]
        self.data[self.pos] = item
        self.pos = self.pos % self.len
        return tmp 
    def __contains__(self, item):    # example algorithm, implements the "in" operator
        if item in self.data:
             return True
        return False
    

class BanditCacheItem(CacheItem):
    def __init__(self, key, item):
        self.key = key
        self.item = item
        self.val = 1     # value of item in list. IT will be selected with probability proportional to val
        self.last_use = -1

class BanditCache(object):
    def __init__(self, capacity, hist_capacity = None, eps = 0.01, eps2 = 0.01, verbose = False):
        self.capacity = capacity 
        if hist_capacity is None:
            hist_capacity = capacity // 2 
        self.hist_capacity = hist_capacity
        self.hash = {}
        self.item_list = []
        self.hist_hash = {}
        self.hist_list = CircularList(self.hist_capacity)
    
        self.eps = eps
        self.eps2 = eps2
        self.iteration = 0  # use for loss function
        self.verbose = verbose
        if verbose:
            self.loads = 0
            self.reuse_rate = 0

    def return_verbose(self):
        return {"loads": self.loads, 'reuse': self.reuse_rate}
    
    def getItem(self, item):
        if np.sum(it.val for it in self.item_list) > 100:
            # print("100")
            for i in range(len(self.item_list)): self.item_list[i].val /= 100
        self.iteration += 1
        # print(self.iteration)
        if item.key in self.hash: # запоминаем когда использовался последний раз и возвращаем то что нужно
            # добавим поощрение за недавнее использование
            loss = 1/(self.iteration - item.last_use)
            loss = loss  * self.eps2
            # item.val =  item.val * np.exp(-loss)  # здесь без минуса так как поощрение
            # print(np.exp(loss), loss)
            self.item_list[self.item_list.index(item)].val = item.val * np.exp(-loss) 
            self.item_list[self.item_list.index(item)].last_use = self.iteration
            return self.item_list[self.item_list.index(item)].item
        else: #select victim and remove it from cache
            if len(self.item_list) >= self.capacity: # select victim between objects in cache
                # print([it.val for it in self.item_list])
                p = np.array([it.val for it in self.item_list])
                p = p/np.sum(p)
                # victim = np.random.choice(self.item_list, 1, p = p)[0]
                # print(victim)
                victim = np.argmax([it.val for it in self.item_list])
                victim = self.item_list[victim]

                self.insertToHist(victim)
                self.removeItem(victim)
                if self.verbose: self.loads += 1  
            # теперь загружаем элемент в с буфер. Для
            # Для элементов из hist используем значения которые сохранились
            # Для элементов не из hist  используем mixin
            # print(len())
            if item.key in self.hist_hash:
                if self.verbose: self.reuse_rate += 1
                item_metadata = self.hist_list.data[self.hist_list.data.index(item)]
                loss = 1 / (self.iteration - item_metadata.last_use)
                loss = loss * self.eps
                print(loss)
                item.val = np.mean([it.key for it in self.item_list]) * np.exp(-loss)
            else:
                if len(self.item_list) > 0:
                    value = np.mean([it.key for it in self.item_list])
                else:
                    value = 1
                item.val = value
            item.last_use = self.iteration
            self.hash[item.key] = item
            self.item_list.append(item)
        return self.item_list[self.item_list.index(item)].item



    def removeItem(self, item):
        del self.hash[item.key]
        del self.item_list[self.item_list.index(item)]

    def insertToHist(self, item):
        removed = self.hist_list.setitem(item)
        if removed is not None:
            del self.hist_hash[removed.key]
        self.hist_hash[item.key] = item
         


