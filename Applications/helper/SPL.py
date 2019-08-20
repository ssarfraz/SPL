import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

class SPL():

    def __init__(self, use_conversion=True):
        # we assume inputs are in -1:1 range. convert these to 0-1 for later yuv conversion. set use_conversion=False if this is not the case
        self.use_conversion = use_conversion


    def GP(self,real, fake):
        real, fake = self.rerange(real,fake)
        real_dx, real_dy = SPL.image_gradients(real)
        fake_dx, fake_dy = SPL.image_gradients(fake)

        rowcol_sim_dx = SPL.get_sim(real_dx, fake_dx)
        rowcol_sim_dy = SPL.get_sim(real_dy, fake_dy)

        GP_loss = rowcol_sim_dx + rowcol_sim_dy

        return GP_loss

    def CP(self,real, fake):
        real, fake = self.rerange(real,fake)
        # rgb
        rgb = SPL.get_sim(real, fake)
        
        #yuv
        real_yuv = SPL.rgb_to_yuv(real)
        fake_yuv = SPL.rgb_to_yuv(fake)
        yuv = SPL.get_sim(real_yuv, fake_yuv)

        ########## grad on color channels
        real_dx, real_dy = SPL.image_gradients(real_yuv)
        fake_dx, fake_dy = SPL.image_gradients(fake_yuv)

        rowcol_sim_dx = SPL.get_sim(real_dx, fake_dx)
        rowcol_sim_dy = SPL.get_sim(real_dy, fake_dy)

        uv_grad = rowcol_sim_dx + rowcol_sim_dy
        #######
        return yuv + uv_grad + rgb

    def rerange(self, real,fake):
        if self.use_conversion:
            real = tf.divide(real + 1, 2)
            fake = tf.divide(fake + 1, 2)
        return real,fake

    def __call__(self, real,fake):        
        gpl = self.GP(real,fake)
        cpl = self.CP(real,fake)

        return gpl+cpl


    @staticmethod
    def get_sim(real, fake):
        # do l2_normlaize if needed
        mat_row = tf.nn.l2_normalize(real, axis=2) * tf.nn.l2_normalize(fake, axis=2)

        in_size = mat_row.get_shape().as_list()

        mat_col = tf.nn.l2_normalize(real, axis=1) * tf.nn.l2_normalize(fake, axis=1)

        ch_sim_row = tf.reduce_sum(tf.reduce_sum(mat_row, axis=2, keep_dims=True))
        ch_sim_col = tf.reduce_sum(tf.reduce_sum(mat_col, axis=1, keep_dims=True))

        sim = -tf.divide(ch_sim_row + ch_sim_col, in_size[1])
        return sim

    ###########################################################

    ## tensorflow functions
    @staticmethod
    def rgb_to_yuv(images):
        """Converts one or more images from RGB to YUV.
        Outputs a tensor of the same shape as the `images` tensor, containing the YUV
        value of the pixels.
        The output is only well defined if the value in images are in [0,1].
        Args:
          images: 2-D or higher rank. Image data to convert. Last dimension must be
          size 3.
        Returns:
          images: tensor with the same shape as `images`.
        """
        _rgb_to_yuv_kernel = [[0.299, -0.14714119,
                               0.61497538], [0.587, -0.28886916, -0.51496512],
                              [0.114, 0.43601035, -0.10001026]]
        images = ops.convert_to_tensor(images, name='images')
        kernel = ops.convert_to_tensor(
            _rgb_to_yuv_kernel, dtype=images.dtype, name='kernel')
        ndims = images.get_shape().ndims
        return math_ops.tensordot(images, kernel, axes=[[ndims - 1], [0]])

    @staticmethod
    def image_gradients(image):
        """Returns image gradients (dy, dx) for each color channel.
        Both output tensors have the same shape as the input: [batch_size, h, w,
        d]. The gradient values are organized so that [I(x+1, y) - I(x, y)] is in
        location (x, y). That means that dy will always have zeros in the last row,
        and dx will always have zeros in the last column.
        Arguments:
          image: Tensor with shape [batch_size, h, w, d].
        Returns:
          Pair of tensors (dy, dx) holding the vertical and horizontal image
          gradients (1-step finite difference).
        Raises:
          ValueError: If `image` is not a 4D tensor.
        """
        if image.get_shape().ndims != 4:
            raise ValueError('image_gradients expects a 4D tensor '
                             '[batch_size, h, w, d], not %s.', image.get_shape())
        image_shape = array_ops.shape(image)
        batch_size, height, width, depth = array_ops.unstack(image_shape)
        dy = image[:, 1:, :, :] - image[:, :-1, :, :]
        dx = image[:, :, 1:, :] - image[:, :, :-1, :]

        # Return tensors with same size as original image by concatenating
        # zeros. Place the gradient [I(x+1,y) - I(x,y)] on the base pixel (x, y).
        shape = array_ops.stack([batch_size, 1, width, depth])
        dy = array_ops.concat([dy, array_ops.zeros(shape, image.dtype)], 1)
        dy = array_ops.reshape(dy, image_shape)

        shape = array_ops.stack([batch_size, height, 1, depth])
        dx = array_ops.concat([dx, array_ops.zeros(shape, image.dtype)], 2)
        dx = array_ops.reshape(dx, image_shape)

        return dx, dy
    ####################################################################################################################