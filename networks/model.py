import tensorflow as tf
import numpy as np

import networks.attention as att


# class ASPP(tf.keras.layers.Layer):
#     def __init__(self, filters):
#         super(ASPP, self).__init__()
#         self.dilation_rates = np.array([(3, 3), (6, 6), (12, 12), (18, 18), (24, 24)])

#         # Dilated Conv 3 x 3
#         self.conv3x3 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', dilation_rate=self.dilation_rates[0])
#         self.bn3x3 = tf.keras.layers.BatchNormalization()
#         self.relu3x3 = tf.keras.layers.ReLU()

#         # Dilated Conv 6 x 6
#         self.conv6x6 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', dilation_rate=self.dilation_rates[1])
#         self.bn6x6 = tf.keras.layers.BatchNormalization()
#         self.relu6x6 = tf.keras.layers.ReLU()

#         # Dilated Conv 12 x 12
#         self.conv12x12 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', dilation_rate=self.dilation_rates[2])
#         self.bn12x12 = tf.keras.layers.BatchNormalization()
#         self.relu12x12 = tf.keras.layers.ReLU()

#         # Dilated Conv 18 x 18
#         self.conv18x18 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', dilation_rate=self.dilation_rates[3])
#         self.bn18x18 = tf.keras.layers.BatchNormalization()
#         self.relu18x18 = tf.keras.layers.ReLU()

#         # Dilated Conv 24 x 24
#         self.conv24x24 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', dilation_rate=self.dilation_rates[4])
#         self.bn24x24 = tf.keras.layers.BatchNormalization()
#         self.relu24x24 = tf.keras.layers.ReLU()

#         self.concat = tf.keras.layers.Concatenate()
#         self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')
#         self.bn = tf.keras.layers.BatchNormalization()
#         self.relu = tf.keras.layers.ReLU()

#     def call(self, input_tensor, **kwargs):
#         x_3 = self.conv3x3(input_tensor)
#         x_3 = self.bn3x3(x_3)
#         x_3 = self.relu3x3(x_3)

#         x_6 = self.conv3x3(input_tensor)
#         x_6 = self.bn3x3(x_6)
#         x_6 = self.relu3x3(x_6)

#         x_12 = self.conv3x3(input_tensor)
#         x_12 = self.bn3x3(x_12)
#         x_12 = self.relu3x3(x_12)

#         x_18 = self.conv3x3(input_tensor)
#         x_18 = self.bn3x3(x_18)
#         x_18 = self.relu3x3(x_18)

#         x_24 = self.conv3x3(input_tensor)
#         x_24 = self.bn3x3(x_24)
#         x_24 = self.relu3x3(x_24)

#         x = self.concat([x_3, x_6, x_12, x_18, x_24])
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(DecoderBlock, self).__init__()
        self.filters = filters

        self.up = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')
        self.conv1 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=3, strides=1, padding='same')
        self.leakyrelu1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.conv2 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=3, strides=1, padding='same')
        self.leakyrelu2 = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, input_tensor, **kwargs):
        x = self.up(input_tensor)
        x = self.conv1(x)
        x = self.leakyrelu1(x)
        x = self.conv2(x)
        x = self.leakyrelu2(x)
        return x


class MVAAutoEncoder():
    def __init__(self):
        self.encoder = tf.keras.applications.DenseNet169(input_shape=(480, 640, 3), include_top=False)
        self.filters = self.encoder.output.shape[-1]

        for layer in self.encoder.layers:
            layer.trainable = True

        # self.aspp = ASPP(int(self.filters / 2))

        self.decoder_block1 = DecoderBlock(int(self.filters / 2))
        self.decoder_block2 = DecoderBlock(int(self.filters / 4))
        self.decoder_block3 = DecoderBlock(int(self.filters / 8))
        self.decoder_block4 = DecoderBlock(int(self.filters / 16))

    def residual_block(self, input_tensor, filters, is_perm=False):
        x = tf.keras.layers.Conv2D(filters=int(filters / 2), kernel_size=3, strides=1, padding='same')(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        
        x = tf.keras.layers.Conv2D(filters=int(filters / 2), kernel_size=3, strides=1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, input_tensor])
        x = tf.nn.relu(x)
        return x

    def multi_view_attention(self, input_tensor, filters, skip=None):
        # _, height_filters, width_filters, channel_filters = input_tensor.shape

        # Channel Attention
        channel = att.serial_connect_attention(input_tensor)

        # Width Attention
        # Transpose (B, H, W, C) → (B, H, C, W) for Width Attention
        width = tf.keras.layers.Permute((1, 3, 2))(input_tensor)
        width = att.serial_connect_attention(width)
        width = tf.keras.layers.Permute((1, 3, 2))(width)

        # Height Attention
        # Transpose (B, H, W, C) → (B, C, W, H) for Height Attention
        height = tf.keras.layers.Permute((3, 2, 1))(input_tensor)
        height = att.serial_connect_attention(height)
        height = tf.keras.layers.Permute((3, 2, 1))(height)

        # x = tf.keras.layers.Add()([channel, width, height])
        # x = tf.keras.layers.Activation('sigmoid')(x)
        # x = tf.keras.layers.Multiply()([x, input_tensor])
        x = tf.keras.layers.Concatenate()([channel, width, height])
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
        x = tf.keras.layers.Activation('sigmoid')(x)
        x = tf.keras.layers.Multiply()([x, input_tensor])

        if skip is not None:
            x = tf.keras.layers.Concatenate()([x, skip])
            x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
        return x

    def net(self):
        # x = self.aspp(self.encoder.output)
        x = self.multi_view_attention(self.encoder.output, self.filters)
             
        # Decoder
        x = self.decoder_block1(x)
        x = self.residual_block(x, int(self.filters / 2))
        x = self.multi_view_attention(x, int(self.filters / 2), self.encoder.get_layer('pool3_pool').output)

        x = self.decoder_block2(x)
        x = self.residual_block(x, int(self.filters / 4))
        x = self.multi_view_attention(x, int(self.filters / 4), self.encoder.get_layer('pool2_pool').output)

        x = self.decoder_block3(x)
        x = self.residual_block(x, int(self.filters / 8))
        x = self.multi_view_attention(x, int(self.filters / 8), self.encoder.get_layer('pool1').output)

        x = self.decoder_block4(x)
        x = self.residual_block(x, int(self.filters / 16))
        x = self.multi_view_attention(x, int(self.filters / 16), self.encoder.get_layer('conv1/relu').output)

        # Last conv
        x = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='same')(x)

        return x

    def build_model(self):
        output_tensor = self.net()
        
        model = tf.keras.Model(inputs=self.encoder.input, outputs=output_tensor, name='AutoEncoder')
        
        return model