import tensorflow as tf
from layerClass import Layer

#tf.compat.v1.disable_v2_behavior()
class Model(Layer,):

    def __init__(self, inputPlaceholder, outputPlaceHolder,isTrainPlaceHolder,learningRate):
        print ("Initialisation")
        self.initializer = "He"
        Layer.__init__(self,self.initializer)
        self.input = inputPlaceholder
        self.output = outputPlaceHolder
        self.learningRate = learningRate
        self._prediction = None
        self._optimize = None
        self._loss = None
        self.isTrain = isTrainPlaceHolder

    def prediction(self):
        print ("Prediction")
        if not self._prediction:
            self._prediction = Layer.runBlock(self, inputData = self.input, isTrain = self.isTrain)
        return self._prediction

    def error(self):
        with tf.name_scope('loss'):
            if not self._loss:
                self._loss = tf.compat.v1.losses.mean_squared_error(labels = self.output, predictions = self._prediction)
            return self._loss


    def optimize(self):
        with tf.name_scope('optimiser'):
            if not self._optimize:
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learningRate)
                update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
                with tf.compat.v1.control_dependencies(update_ops):
                    self._optimize = optimizer.minimize(self._loss)
                #self.train_op = tf.group([self._optimize, update_ops])
            return self._optimize
