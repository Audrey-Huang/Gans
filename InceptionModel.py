import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
from cleverhans.model import Model
slim = tf.contrib.slim
tensorflow_master = ""
checkpoint_path = "./inception_v3.ckpt"
inputimg="./carplate/1.jpg"
output_dir = "./outimage"
max_epsilon = 4.0
image_width = 299
image_height = 299
batch_size = 50
import sys
sys.path.append('./cleverhans')
eps = 5* max_epsilon / 255.0
batch_shape = [batch_size, image_height, image_width, 3]
nb_classes = 1001

class InceptionModel(Model):

    def __init__(self, nb_classes):
        super(InceptionModel,self).__init__(nb_classes=nb_classes,needs_dummy_fprop=True)
        self.built = False

    def __call__(self, x_input,return_logits=False):
        """Constructs model and return probabilities for given input."""

        reuse = True if self.built else None

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(

                x_input, num_classes=self.nb_classes, is_training=False,

                reuse=reuse)

        self.built = True
        self.logits = end_points['Logits']
        self.probs = end_points['Predictions'].op.inputs[0]
        if return_logits:
            return self.logits
        else:
            return self.probs

    def get_logits(self, x_input):
        return self(x_input, return_logits=True)

    def get_probs(self, x_input):
        return self(x_input)