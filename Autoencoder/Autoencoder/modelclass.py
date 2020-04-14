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
        #self.probabilityPlaceholder = probabilityPlaceholder

    def prediction(self):
        print ("Prediction")
        if not self._prediction:
            self._prediction = Layer.runBlock(self, inputData = self.input, isTrain = self.isTrain)
        return self._prediction

    def error(self):
        with tf.name_scope('loss'):
            if not self._loss:
                '''
                numerator = 2 * tf.reduce_sum(self.output * tf.sigmoid(self._prediction), axis=(1,2,3))
                denominator = tf.reduce_sum(tf.square(self.output) + tf.square(tf.sigmoid(self._prediction)), axis=(1,2,3))
                epsilon = 1e-6
                self.dice =  tf.reduce_mean(tf.reshape( numerator / (denominator+epsilon), (-1, 1, 1, 1)))
                self.diceloss = 1 - self.dice
                self.CE = tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(labels = self.output, logits = self._prediction)
		        #self.loss1 = tf.nn.weighted_cross_entropy_with_logits(targets=self.output,logits=self._prediction,pos_weight=0.9)
                self._loss = tf.reduce_mean(self.CE) + self.diceloss
                '''
                self._loss = tf.compat.v1.losses.mean_squared_error(labels = self.output, predictions = self._prediction)
                #self._loss = self.diceloss
            return self._loss


    def optimize(self):
        with tf.name_scope('optimiser'):
            if not self._optimize:
                #optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learningRate, momentum = 0.9)
                #optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate = self.learningRate, momentum = 0.9, use_nesterov=False)
                #optimizer = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learningRate)
                update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
                with tf.compat.v1.control_dependencies(update_ops):
                    self._optimize = optimizer.minimize(self._loss)
                #self.train_op = tf.group([self._optimize, update_ops])
            return self._optimize
