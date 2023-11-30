import numpy as np

class Param_search:
    def __init__(self, iters):
        self.iters_to_trigger = iters
        self.iter = 0
        self.stat = None
        self.prev_step_sign = None
    # на данный этап захардкодим параметры
        self.betas_len = 30
        self.gammas_len = 30
        self.gammas = np.logspace( np.log10(0.000001), np.log10(0.2), self.gammas_len)
        self.betas = np.logspace(np.log10(1.01), np.log10(1.3), self.betas_len)
    def init_params(self,):
        self.beta = np.random.choice(range(self.betas_len- 1))
        self.gamma = np.random.choice(range(self.gammas_len- 1))
        return self.gammas[self.gamma], self.betas[self.beta]
    def __call__(self,iter, stat,):
        if self.iter > self.iters_to_trigger:
            if self.stat is None:
                self.prev_beta = self.beta
                self.prev_gamma = self.gamma
                self.beta += 1
            else:
                delta = (stat - self.stat)
                tmp = np.random.rand()
                if tmp > 0.5:
                    delta = delta * (self.beta - self.prev_beta)
                    # delta_beta =  1 - 2 *(delta< 0)
                    self.prev_beta = self.beta
                    self.beta = max(0, min( self.betas_len - 1, self.beta + 1 - 2 *(delta< 0)))
                else:
                    delta = delta * (self.gamma - self.prev_gamma)
                    self.prev_gamma = self.gamma
                    self.gamma =  max(0, min( self.gammas_len - 1, self.gamma +1 - 2* (delta< 0)))
                # print(self.gammas[self.gamma], self.betas[self.beta])
                # print(f"            delta: {delta}")
            self.stat = stat
            stat = 0
            self.iter = 0
        else:
            self.iter += 1
        return self.gammas[self.gamma], self.betas[self.beta], stat

def cache_obs(requests, K, P, N):
    miss = 0
    cache = [None for i in range(K)]
    for pos in range(len(requests)):
        observations = requests[pos: min(pos + P, len(requests))]
        req = requests[pos]
        # print(cache, req)
        if req in cache:
            cache.remove(req)
            cache.append(req)
        else:
            if None in cache:
                del cache[0]
            else:
                C_obs = set(cache) - set(observations)
                if len(C_obs) != 0: # есть элемент для удаления
                    ind = 0
                    while not (cache[ind] in C_obs):
                        ind += 1
                    # нашли индекс для удаления
                    del cache[ind]
                else: # все элементы в кэше запрошены на будущее. удаляем то, которое запрошено на более позднее время
                    _key = None
                    mmin = -np.inf
                    for key in cache: 
                        try:
                            ind = observations.index(key)
                        except:
                            ind = np.inf  #not found
                        if ind > mmin:
                            mmin = ind
                            _key = key
                    
                    cache.remove(_key)
            cache.append(req)
            miss += 1
    return miss / len(requests)

from collections import defaultdict

from collections import defaultdict
def cache_prob_mul(requests, K, N, gamma = 0.001, beta = 1.2):
    eps = 0.01

    def normalize0(func, w):
        return func
        return (1 - func) * ((1- eps - w) / (1 - 2 * eps))**4 + func
    def normalize1(func, w):
        return func
        return (1 - func)* ((w- eps) / (1 - 2 * eps))**4 + func 

    # gamma = 0.001
    # beta = 1.2
    miss = 0
    cache = [None for i in range(K)]
    cache_weights = defaultdict(lambda : 0.5)
    weights = defaultdict(lambda : 0.5)
    times = defaultdict(lambda : 0.)   # time from last need
    p_for_step = []
    for pos in range(len(requests)):
        req = requests[pos]
        # print(cache, req)
        if req not in times.keys():
            times[req] = pos
        weights[req] =  max(eps, weights[req] * normalize0((1 - gamma), weights[req]) ** (pos - times[req]))
        weights[req]  = min(1 - eps, weights[req] * normalize1(beta, weights[req]))  # update weight of page
        times[req] = pos
        if req in cache:
            cache.remove(req)
            cache.append(req)
            cache_weights[req] = weights[req]
        else:
            if None in cache:
                del cache[0]
            else:
                p_s = []
                weight_mean = np.mean(list(weights.values())) #np.percentile(list(weights.values()), 50)
                while True:
                    item = cache[0]
                    weights[item] = weights[item] * normalize0((1 - gamma), weights[item]) ** (pos - times[item])
                    weights[item] = max(eps, min(1 - eps, weights[item]))
                    times[item] = pos
                    weight_cache = weights[item] 
                    weight_req = weights[req]
                    
                    # there implement function for combining weights
                    # 
                    pow = 3.1
                    p = np.random.beta(weight_cache + 0.5, weight_req + 6)

                    eps0 = 0.
                    # p = max(0.1, min(0.6, p))
                    p_s.append(p)
                    survive = False # np.random.choice([False, True], p = [1 - p, p])
                    item = min(cache_weights.items(), key = lambda x: x[1])[0]
                    if survive:
                        del cache[0]
                        cache.append(item)
                    else:
                        cache.remove(item)
                        del cache_weights[item]
                        break
                p_for_step.append(np.mean(p_s))
            cache.append(req)
            cache_weights[req] = weights[req]
            miss += 1
    # update all weights
    for it in weights.keys():
        weights[it] = weights[it] * ((1 - gamma) ** (len(requests) - times[item]))
        
    return miss / len(requests), weights, p_for_step

def cache_prob_mul_separate(requests, K, N, gamma = 0.001, beta = 1.2):
    param_search = Param_search(K * 30)
    eps = 0.05
    gamma, beta = param_search.init_params()
    miss = 0
    hits = 0
    cache = [None for i in range(K)]
    weights = defaultdict(lambda : 0.5)
    times = defaultdict(lambda : 0.)   # time from last need
    p_for_step = []
    for pos in range(len(requests)):
        req = requests[pos]
        # print(cache, req)
        if req not in times.keys():
            times[req] = pos
        weights[req] = max(eps, weights[req] * (1 - gamma) ** (pos - times[req]))
        weights[req]  = min(1. - eps,weights[req]* beta)  # update weight of page
        times[req] = pos
        
        if req in cache:
            cache.remove(req)
            cache.append(req)
            hits += 1
        else:
            if None in cache:
                del cache[0]
            else:
                p_s = []
                while True:
                    item = cache[0]
                    weights[item] = max(eps, weights[item] * ((1 - gamma) ** (pos - times[item])))
                    times[item] = pos
                    weight_cache = weights[item] 
                    weight_req = weights[req]
                    
                    # there implement function for combining weights
                    # 
                    # p = weight_cache / max(weight_cache,weight_req ) # ((weight_cache**3 + weight_req**3)** (1/3))
                    p = weight_cache
                    p = max(0.05, min(1 - 0.05, p))
                    p_s.append(p)
                    survive = np.random.choice([False, True], p = [1 - p, p])
                    if survive:
                        del cache[0]
                        cache.append(item)
                    else:
                        del cache[0]
                        break
                p_for_step.append(np.mean(p_s))
            cache.append(req)
            miss += 1
        gamma, beta, hits= param_search(pos, hits)
        # hits = hits if pos % (K * 5) == 0 else 0
    # update all weights
    for it in weights.keys():
        weights[it] = weights[it] * ((1 - gamma) ** (len(requests) - times[item]))
        
    return miss / len(requests), weights, p_for_step


def cache_prob(requests, K, N, gamma = 0.01, beta = 0.1):
    eps = 0.1
    # gamma = 0.0001
    # beta = 0.13
    miss = 0
    cache = [None for i in range(K)]
    weights = defaultdict(lambda : 0.5)
    times = defaultdict(lambda : 0.)   # time from last need
    for pos in range(len(requests)):
        req = requests[pos]
        # print(cache, req)
        if req not in times.keys():
            times[req] = pos
        weights[req] = max(eps, weights[req] * (1 - gamma) ** (pos - times[req]))
        weights[req]  = min(1. - eps,  weights[req]  + beta)  # update weight of page
        times[req] = pos
        
        if req in cache:
            cache.remove(req)
            cache.append(req)
        else:
            if None in cache:
                del cache[0]
            else:
                while True:
                    item = cache[0]
                    weights[item] = max(eps, weights[item] * (1 - gamma) ** (pos - times[item]))
                    times[item] = pos
                    weight = weights[item] 
                    p = weight #(weight - np.min(list(weights.values()))) / (weight + weights[req] -  2 * np.min(list(weights.values())))
                   
                    survive = np.random.choice([False, True], p = [1 - p, p])
                    if survive:
                        del cache[0]
                        cache.append(item)
                    else:
                        del cache[0]
                        break
            cache.append(req)
            # weights[req] = 0.5
            miss += 1
        # renovate weight
    
    return miss / len(requests), weights



def cache_fixed_prob(requests, K, N, pow = 1., gamma = 0.001):
    eps = 0.1
    # beta = 0.13
    miss = 0
    cache = [None for i in range(K)]
    weights = np.ones((N,), dtype=float)/ N
    times = defaultdict(lambda : 0.) 
    for pos in range(len(requests)):
        req = requests[pos]
        weights[req] = gamma + (1 - gamma) ** (pos - times[req]) * weights[req]
        times[req] = pos
        # times[req] = pos
        if req in cache:
            cache.remove(req)
            cache.append(req)
        else:
            if None in cache:
                del cache[0]
            else:
                while True:
                    item = cache[0]
                    weights[item] = (1 - gamma) ** (pos - times[item]) * weights[item] # предполагаем что полный круг делается доволльно редко
                    times[item] = pos
                    w2 = weights[item] 
                    w1 = weights[req]
                    p = (w2 ** pow)/ (w1**pow + w2**pow) #(weight - np.min(list(weights.values()))) / (weight + weights[req] -  2 * np.min(list(weights.values())))
                   
                    survive = np.random.choice([False, True], p = [1 - p, p])
                    if survive:
                        del cache[0]
                        cache.append(item)
                    else:
                        del cache[0]
                        break
            cache.append(req)
            # weights[req] = 0.5
            miss += 1
        # renovate weight
    
    return miss / len(requests), weights