from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.keras as keras
from tensorflow.python.keras import backend as K
import tensorflow as tf

class ncs():
    def __init__():
        # nothing to say now
        pass
    
    def to_ncs_graph(model_file, weight_file):
        config = None
        with open(model_file, "r") as file:
            config = file.read()
        
        K.set_learning_phase(0)
        model = keras.models.model_from_json(config)
        model.load_weights(weight_file)
        saver = tf.train.Saver()
        sess = K.get_session()
        saver.save(sess, "./TF_Model/tf_model")
        fw = tf.summary.FileWriter('logs', sess.graph)
        fw.close()
