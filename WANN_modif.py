# from keras.activations import *
import random
import numpy as np
from keras.losses import mean_squared_error as MSE
from keras.losses import mean_absolute_error as MAE
from keras.losses import mean_absolute_percentage_error as MAPE
import tensorflow as tf

from typing import List
from math import e
from math import tanh
from math import exp
from math import sin
from math import cos
"""# WANN

### Constants
"""

NOT_INPUT_LAYER_FEED_ERROR = '''
Tried to feed input vec to hidden or output layer.
Expected input layer.
'''

OUTPUT_INSERTION = '''
Insertion to the output layer
'''

ERROR = '''
Layer type not found
'''
losses = {'MSE': MSE, 'MAE': MAE, 'MAPE': MAPE}


def relu(x: float):
    return max(0., x)


def linear(x: float):
    return x if x >= 0 else 2*x


def gauss(x: float):
    return e**(x**(-2))


def sigmoid(x: float):
    return 1/(1+e**(-x))


random.seed(146)

func_dict = {
    'relu': relu,
    'lin': linear,
    'sigm': sigmoid,
    'tanh': tanh,
    'exp': exp,
    'gauss': gauss,
    'sin': sin,
    'cos': cos
}


func_list = list(func_dict.keys())


NN_DEBUG_MSG1 = ''' 
Layer = {}
    nodes_id    =   {}    
'''
NN_DEBUG_MSG2 = '''
    children_idx:
'''
NN_DEBUG_MSG3 = '''
    nodes_vals  =   {}
    nodes_funcs =   {}
'''

"""### Input"""

'''############ Code bellow is deprecated to use! ###############'''
'''############ May cause Undefined Behaviour!    ###############'''


class InputNode:
    def __init__(self, parents_idx, value=0.):
        self.parents_idx = parents_idx
        self.value = value
        self.node_type = 'input'
        self.function = 'id'

    def receive(self, value):
        self.value = value

    def forward(self, weight):
        return self.value * weight

    def add_parent(self, parent_id):
        self.parents_idx.append(parent_id)


class InputLayer:
    def __init__(self, nodes_id):
        self.nodes = nodes_id
        self.level = 0
        self.layer_type = 'input'

    def receive(self, nodes_set, x):
        for x, node_id in zip(x, self.nodes):
            nodes_set[node_id].receive(x)

    def forward(self, nodes_set, weight):
        for nid in self.nodes:
            val_to_pass = nodes_set[nid].forward(weight)
            for pid in nodes_set[nid].parents_idx:
                nodes_set[pid].receive(val_to_pass)


"""### Hidden"""


class HiddenNode:
    def __init__(self, parents_idx, function_name=random.choice(func_list), value=0.):
        self.parents_idx = parents_idx
        self.value = value
        self.function = func_dict[function_name]
        self.node_type = 'hidden'

    def receive(self, value):
        self.value += value

    def forward(self, weight):
        self.value = self.function(self.value)
        return self.value * weight

    def add_parent(self, parent_id):
        self.parents_idx.append(parent_id)


class HiddenLayer:
    def __init__(self, nodes_id, level):
        self.nodes = nodes_id
        self.level = level
        self.layer_type = 'hidden'

    def forward(self, nodes_set, weight):
        for nid in self.nodes:
            val_to_pass = nodes_set[nid].forward(weight)
            for pid in nodes_set[nid].parents_idx:
                nodes_set[pid].receive(val_to_pass)

    def insert_node(self, new_node_id):
        self.nodes.append(new_node_id)


class OutputNode:
    def __init__(self, value=0.):
        self.value = value
        self.function = 'id'
        self.parents_idx = [None]
        self.node_type = 'output'

    def receive(self, value):
        self.value += value


class OutputLayer:
    def __init__(self, nodes_id):
        self.nodes = nodes_id
        self.level = -1
        self.layer_type = 'output'

    def forward(self, nodes_set):
        out = []
        for nid in self.nodes:
            out.append(nodes_set[nid].value)
        return out


'''############ Code above is deprecated to use!    ###############'''
'''############ Use Node and Layer classes instead. ###############'''


class Node:
    def __init__(self, children_idx: List[int], node_type: str, value=0., function_name=None):
        self.children_idx = children_idx
        self.value = value
        self.node_type = node_type
        if function_name is None:
            function_name = random.choice(func_list)
        self.function_name = function_name
        self.function = func_dict[function_name] if node_type == 'hidden' else 'id'

    def receive(self, value):
        if self.node_type == 'input':
            self.value = value
        else:
            self.value += value

    def forward(self, weight):
        if self.node_type == 'input':
            res = self.value * weight
            self.value = 0.
            return res
        if self.node_type == 'hidden':
            res = self.function(self.value) * weight
            self.value = 0.
            return res
        if self.node_type == 'output':
            res = self.value
            self.value = 0.
            return res
        raise ERROR

    # TODO deprecated
    def add_children(self, parent_id):
        self.children_idx.append(parent_id)


class Layer:
    def __init__(self, nodes_id: List[int], layer_type: str):
        self.nodes = nodes_id
        self.layer_type = layer_type

    def feed(self, nodes_set: List[Node], x: List[float]):
        if self.layer_type == 'input':
            for x, node_id in zip(x, self.nodes):
                nodes_set[node_id].receive(x)
        else:
            raise NOT_INPUT_LAYER_FEED_ERROR

    def forward(self, nodes_set: List[Node], weight: float):
        if self.layer_type != 'output':
            vals = []
            for nid in self.nodes:
                val_to_pass = nodes_set[nid].forward(weight)
                for cid in nodes_set[nid].children_idx:
                    nodes_set[cid].receive(val_to_pass)
                vals.append(val_to_pass)

            return vals
        else:
            out = []
            for nid in self.nodes:
                out.append(nodes_set[nid].forward(weight))
            return out


"""### Neural Net ###"""


class NN:
    def __init__(self, input_size: int, output_size: int):
        self.nodes_set = []
        self.input_size = input_size
        self.output_size = output_size

        # Creating input nodes

        for _ in range(input_size):
            out_idx = list(range(input_size, input_size + output_size))
            cids = random.sample(out_idx,
                                 k=random.randint(1, output_size))
            # print('{} node\'s cids = {}'.format(len(self.nodes_set) + 1, cids))
            self.nodes_set.append(Node(cids, node_type='input'))

        # Creating output nodes:

        for i in range(output_size):
            self.nodes_set.append(Node([], node_type='output'))

        # Add new layers
        self.layers = [Layer(list(range(input_size)), layer_type='input'),
                       Layer(list(range(input_size, input_size + output_size)), layer_type='output')]

        # For Genetic Alg
        self.loss = 0

    def find_layer_num(self, node_id):
        for l in self.layers:
            if node_id in l.nodes:
                print('l =', self.layers.index(l), l)
                return self.layers.index(l)
        print('! Layer not found, node_id ={}'.format(node_id))
        return 'error'

    def rand_conn(self):
        pid = random.randint(0, len(self.nodes_set) - 1)
        while self.nodes_set[pid].node_type == 'output' or len(self.nodes_set[pid].children_idx) == 0:
            pid = random.randint(0, len(self.nodes_set) - 1)
        # print(len(self.nodes_set[pid].children_idx))
        cid = random.choice(self.nodes_set[pid].children_idx)
        return pid, cid

    def cut_conn(self):
        # Cut random connection
        pid, cid = self.rand_conn()

        # Create and add new node
        new_node = Node([cid], node_type='hidden')
        self.nodes_set.append(new_node)
        new_node_id = len(self.nodes_set)-1

        # Connect new node to parental node
        # TODO -- genomes
        self.nodes_set[pid].children_idx.remove(cid)
        self.nodes_set[pid].children_idx.append(new_node_id)

        # Find number of layers of parental and children node
        cid_layer_num = self.find_layer_num(cid)
        pid_layer_num = self.find_layer_num(pid)
        print('cid layer =', cid_layer_num)
        print('pid layer =', pid_layer_num)

        # Add node or layer
        if cid_layer_num - pid_layer_num == 1:
            # If there is no layers between cid and pid layers
            # Create new Layer and add it
            new_layer = Layer([new_node_id], layer_type='hidden')
            self.layers.insert(pid_layer_num + 1, new_layer)
            print('Added new layer {} with node {} between {} and {}'.format(pid_layer_num + 1, new_node_id, pid, cid))
        else:
            # If there are another layer(s) between
            # add new node randomly into one of the layers
            layer_to_insert = random.randint(1, len(self.layers) - 2)
            print(len(self.layers) - 2)
            if self.layers[layer_to_insert].layer_type not in ['output', 'input']:
                self.layers[layer_to_insert].nodes.append(new_node_id)
                print('Added new node {} into {} layer'.format(new_node_id, layer_to_insert))
            else:
                print(OUTPUT_INSERTION)

    # def add_conn(self, pid):
    #     pid = random.choice(len(self.nodes_set))
    #     cid = random.choice(len(self.nodes_set))
    #     while find_layer_num()

    # def add_layer(self):
    #     new_node = HiddenNode([])
    #     new_layer = HiddenLayer()

    def change_activation(self):
        change_node_id = random.randint(self.input_size + self.output_size - 1, len(self.nodes_set) - 1)
        function_name = random.choice(func_list)

        if self.nodes_set[change_node_id].node_type == 'input' or self.nodes_set[change_node_id].node_type == 'output':
            print('!FATAL ERROR: Trying to change input/output node activation function')
        else:
            print('Activation function of {} node changed from {} to {}'.format(change_node_id,
                                                                                self.nodes_set[change_node_id].function_name,
                                                                                function_name))
            self.nodes_set[change_node_id].function = func_dict[function_name]

    def mutate(self):
        action = random.choice(['cut', 'change'])
        if action == 'cut':
            self.cut_conn()
        if action == 'change':
            self.change_activation()

    def feed(self, x, weight=1.):
        self.layers[0].feed(self.nodes_set, x)
        res = []
        for layer in self.layers:
            res = layer.forward(self.nodes_set, weight)
            # print(res)
        return np.array(res)

    def print(self):
        for layer in self.layers:
            # print(layer.nodes)
            # print(layer)
            print(NN_DEBUG_MSG1.format(layer.layer_type,
                                       layer.nodes))
            print(NN_DEBUG_MSG2, end='')
            for i in layer.nodes:
                print(' '*8, self.nodes_set[i].children_idx)

            print(NN_DEBUG_MSG3.format([self.nodes_set[i].value for i in layer.nodes],
                                       [self.nodes_set[i].function for i in layer.nodes]))

    def pprint(self):
        for layer in self.layers:
            print(self.layers.index(layer), layer.nodes)


class NaiveGenetic:
    def __init__(self, pop_size: int, epoch_num: int, x: np.array, y: np.array, loss_fn_name: str):
        self.pop_size = pop_size
        self.epochs = epoch_num
        self.x = x
        self.y = y
        if loss_fn_name not in losses.keys():
            print('Fatal Error: Incorrect Loss Function')
        self.loss = losses[loss_fn_name]
        self.population = []
        if x.shape[0] != y.shape[0]:
            print('!Fatal error: x dim={} =/= y dim={}'.format(x.shape[0], y.shape[0]))
        self.lines_num = x.shape[0]

        if len(x.shape) != 1:
            self.input_size = x.shape[1]
        else:
            self.input_size = 1
        if len(y.shape) != 1:
            self.output_size = y.shape[1]
        else:
            self.output_size = 1
        for i in range(self.pop_size):
            self.population.append(NN(self.input_size, self.output_size))
        self.chart = []

    def run(self):
        best = []
        for epoch in range(self.epochs):
            print('Epoch 1 started executing')
            for nn in self.population:
                for i in range(self.lines_num):
                    x_pred = nn.feed(self.x[i], 1)
                    nn.loss = self.loss(x_pred, self.y[i])
                    self.chart.append(nn)
            print('Dataset ran')

            self.chart = sorted(self.chart, key=lambda n: tf.dtypes.cast(n.loss,  tf.float32))
            best = self.chart[:self.pop_size//2]

            # Mutation
            for nn in best:
                nn.mutate()

            print('Epoch', epoch, 'loss =', best[0].loss)
        return best


class WANN(NaiveGenetic):
    def __init__(self, pop_size: int, epoch_num: int, x: np.ndarray, y:np.array):
        self.pop_size = pop_size
        self.epochs = epoch_num

class iWANN(WANN):
    pass
