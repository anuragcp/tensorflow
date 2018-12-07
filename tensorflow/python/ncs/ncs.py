from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras as import keras
from tensorflow.python.keras import backend as K
import tensorflow

class ncs():
    def __init__(self):
        # nothing to say now
        pass
    
    def to_ncs_graph(self, model_file, weight_file):
        self.model_file = model_file
        self.weight_file = weight_file
        self.config = None
        with open(self.model_file, "r") as file:
            self.config = file.read()
        
        K.set_learning_phase(0)
        self.model = keras.models.model_from_json(self.config)
        self.model.load_weights(self.weight_file)
        self.saver = tensorflow.python.train.Saver()
        self.sess = K.get_session()
        self.saver.save(self.sess, "./TF_Model/tf_model")
        self.fw = tensorflow.python.summary.FileWriter('logs', self.sess.graph)
        self.fw.close()
