import torch
import numpy as np
import torch.nn.functional as F
from scipy.stats import norm
def generate(n_objects = 10, 
            length = 100, 
            random_seed = 0):
    np.random.seed(random_seed)
    # P = np.random.rand(n_objects, n_objects) * 1.5
    # P = torch.tensor(P)
    # P =F.softmax(P, dim = 1).numpy()  # row is probability distribution
    P = np.random.normal(loc = 1, scale= 2, size = (n_objects, n_objects))
    P = torch.tensor(P)
    P =F.softmax(P, dim = 1).numpy()  # row is probability distribution

    page = np.random.choice(range(n_objects))
    for i in range(length):
        page = np.random.choice(range(n_objects), p = P[page])
        yield page

def generate_zipf(lenght = 100, param = 1.5, random_seed = 0):
    np.random.seed(random_seed)
    pages = np.random.zipf(a = param, size = lenght)
    return pages

def generate_zipf_trend(lenght = 100, 
                        period = 10, 
                        param = 1.5, 
                        random_seed = 0):
    shift = 0
    np.random.seed(random_seed)
    for i in range(1, lenght + 1):
        page = np.random.zipf(a = param) + shift
        yield page
        if i % period == 0:
            shift += 10

def generate_with_shift(n_objects = 10,
                        length = 100,
                        shift_time = 1000, 
                        disperse = 1, 
                        eps = 1e-2, 
                        random_seed = 0):
    """
    генерация запросов со сдвигом по времени. 
    Берется нормальное распределение как профиль вероятностей запросов к файлам. По нему генерируется запрос, 
    после генерации распределение сдвигается так, чтобы наиболее востребованный файл за время shift_time был в конце
    
    Также с вероятностью eps файл генерируется не из профиля, а из равномерного распределения

    n_objects: число объектов в кеше. 
    length:    длительность генерируемой последовательности
    shift_time: время за которое сдвигается профиль
    disperce: ширина профиля. дается в долях от количества объектов
    """
    np.random.seed(random_seed)
    assert 0.0 < disperse < 1. , f'disperce must be in range [0, 1] but get {disperse}'
    objects = np.array(list(range(0, n_objects )))
    mu  = 0.
    delta = n_objects/ shift_time

    def get_probs():
        probs = norm.pdf(x = objects,loc = mu, scale = disperse * n_objects)
        np.divide(probs, np.sum(probs), out = probs)
        return probs

    done = False
    for i in range(length):
        if (i > length //2) and not done:
            disperse *= disperse
            done = True
        greedy = np.random.rand() < eps
        if greedy:
            request = np.random.choice(objects)
        else:
            probs = get_probs()
            request = np.random.choice(objects, p = probs)
            mu = mu + delta
            if mu > n_objects:
                mu = 0

        yield request

def generate_distribution(n_objects = 10, 
            length = 100, 
            random_seed = 0):
    np.random.seed(random_seed)
    p_s = np.random.rand(n_objects) ** 0.5  #+ 1000
    p_s = p_s / np.sum(p_s)
    pages = np.random.choice(n_objects, size = length, p = p_s)
    return pages, p_s