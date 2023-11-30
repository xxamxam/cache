from datetime import datetime
from src.utils import CacheItem
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Norm(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
    def forward(self, x):
        return x/torch.norm(x, 'fro')

class Model(nn.Module):
    def __init__(self, feature_num, emb_dim = 10):
        super(self.__class__, self).__init__()
        self.features = nn.Sequential(
            nn.Embedding(feature_num, emb_dim),
            Norm()
        )
    def forward(self, left, right = None):
        left_emb = self.features(left) #.detach()
        if right is None:
            return left_emb
        else:
            right_emb = self.features(right)
            loss = torch.sum(left_emb * right_emb)
            return right_emb.detach(), -loss


class NetCacheItem(CacheItem):
    """Data structure of items stored in cache"""
    def __init__(self, key, item):
        self.key = torch.IntTensor([key])
        self.item = item
        self.vec = torch.Tensor()


class NetCache(object):
    """A sample class that implements LRU algorithm"""

    def __init__(self, length, cache_items = 1000, emb_dim = 10, first_item = 0, verbose = False):
        self.length = length
        self.hash = {}
        self.item_list = []
        self.verbose = verbose
        if verbose:
            self.loads = 0
        self.prev_item_key = torch.IntTensor([first_item])
        self.model = Model(cache_items, emb_dim)
        self.optim = torch.optim.SGD(self.model.parameters(), lr = 3e-2)

    def return_verbose(self):
        return {"loads": self.loads}
    
    def getItem(self, item):
        """Insert new items to cache"""
        item_vec, loss = self.model(self.prev_item_key, item.key)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.prev_item_key = item.key
        item.vec = item_vec


        if item.key in self.hash:
            # Move the existing item to the head of item_list.
            return None
        else:
            # Remove the last item if the length of cache exceeds the upper bound.
            if len(self.hash) >= self.length:
                if self.verbose: self.loads += 1   # Пока что тупой, но все же какой-то контроль над тем что выкидывается
                # select item to be deleted
                meann = torch.mean(torch.stack([it[1].vec for it in self.hash.items()]+ [item_vec]))
                item_to_delete = min(self.hash.items(), key=lambda x: torch.sum(meann * x[1].vec))
                self.removeItem(item_to_delete[1])

            # If this is a new item, just append it to
            # the front of item_list.
            self.hash[item.key] = item

    def removeItem(self, item):
        """Remove those invalid items"""
        del self.hash[item.key]