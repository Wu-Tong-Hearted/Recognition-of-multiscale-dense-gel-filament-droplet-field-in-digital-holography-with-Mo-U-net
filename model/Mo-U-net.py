import tensorflow as tf
from tensorflow.keras import Model

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result


class UnetTiny(Model):
    def __init__(self, classes=10, input_shape=[224*3, 224*3, 3], trainable=True):
        super(UnetTiny, self).__init__()
        self.classes = classes
        self.__name__ = 'UnetTiny'

        # base_model = tf.keras.applications.MobileNetV3Large(input_shape=input_shape, include_top=False)
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)

        # 使用这些层的激活设置
        layer_names = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',      # 4x4
        ]
        layers = [base_model.get_layer(name).output for name in layer_names]

        # 创建特征提取模型
        self.down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

        self.down_stack.trainable = trainable

        self.up_stack = [
            upsample(512, 3),  # 4x4 -> 8x8
            upsample(256, 3),  # 8x8 -> 16x16
            upsample(128, 3),  # 16x16 -> 32x32
            upsample(64, 3),   # 32x32 -> 64x64
        ]
        self.last = tf.keras.layers.Conv2DTranspose(self.classes, 3, strides=2, padding='same')
        self.concat = tf.keras.layers.Concatenate()
        #self.inputs = tf.keras.layers.Input(shape=self.input_shape)
        self.true_last = tf.keras.layers.Conv2D(self.classes, 3, strides=1, padding='same', activation='softmax')

    def call(self, x):
        # 在模型中降频取样
        skips = self.down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # 升频取样然后建立跳跃连接
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = self.concat([x, skip])
        # 这是模型的最后一
        x = self.last(x)
        x = self.true_last(x)
        return x



if __name__ == '__main__':
    OUTPUT_CHANNELS=2
    model = UnetTiny(classes=OUTPUT_CHANNELS)
    model.build(input_shape=[2,224*3,224*3,3])
    model.summary()
    # model.compile(optimizer='adam',
    #             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #             metrics=['accuracy'])