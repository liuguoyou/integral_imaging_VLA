import tensorflow as tf
import numpy as np
import os

from depth_networks.model import MVAAutoEncoder


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'


class DepthEstimation(object):
    def __init__(self):
        pass
        
    def preprocess(self, rgb_path):
        raw_rgb = tf.io.read_file(rgb_path)
        decoded_rgb = tf.image.decode_jpeg(raw_rgb)
        rgb = tf.image.convert_image_dtype(decoded_rgb, dtype=tf.float32)
        rgb = tf.expand_dims(rgb, axis=0)
        return rgb
    
    def load_model(self, model_path):
        net = MVAAutoEncoder()
        model = net.build_model()
        model.load_weights(model_path)
        return model

    def prediction(self, rgb_path, model_path):
        rgb = self.preprocess(rgb_path)
        model = self.load_model(model_path)

        pred = model.predict(rgb)
        pred = pred.reshape((240, 320))
        pred = np.clip((1000 / pred), 10, 1000)
        
        from skimage.transform import resize
        pred = resize(pred, (480, 640), order=1, preserve_range=True, mode='reflect', anti_aliasing=True)
        return pred