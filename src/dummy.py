from datetime import datetime
from src.utils import CacheItem
import numpy as np
class LRUCacheItem(CacheItem):
    """Data structure of items stored in cache"""
    def __init__(self, key, item):
        self.key = key
        self.item = item
        self.timestamp = datetime.now()


class DummyCache(object):
    """A sample class that implements LRU algorithm"""

    def __init__(self, length, delta=None, verbose = False):
        self.length = length
        self.delta = delta
        self.hash = {}
        self.item_list = []
        self.verbose = verbose
        if verbose:
            self.loads = 0

    def return_verbose(self):
        return {"loads": self.loads}
    
    def getItem(self, item):
        """Insert new items to cache"""

        if item.key in self.hash:
            return 
        else:
            # Remove the last item if the length of cache exceeds the upper bound.
            if len(self.item_list) > self.length:
                if self.verbose: self.loads += 1   # Пока что тупой, но все же какой-то контроль над тем что выкидывается
                ind = np.random.choice(self.item_list, 1)[0]
                self.removeItem(ind)

            # If this is a new item, just append it to
            # the front of item_list.
            self.hash[item.key] = item
            self.item_list.insert(0, item)

    def removeItem(self, item):
        """Remove those invalid items"""

        del self.hash[item.key]
        del self.item_list[self.item_list.index(item)]

    def validateItem(self):
        """Check if the items are still valid."""

        def _outdated_items():
            now = datetime.now()
            for item in self.item_list:
                time_delta = now - item.timestamp
                if time_delta.seconds > self.delta:
                    yield item
        map(lambda x: self.removeItem(x), _outdated_items())