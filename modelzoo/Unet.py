__author__ = 'Fabian Isensee'
from collections import OrderedDict
from lasagne.layers import InputLayer, ConcatLayer, Pool2DLayer, ReshapeLayer, DimshuffleLayer, NonlinearityLayer, DropoutLayer, Upscale2DLayer, batch_norm
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
import lasagne


def build_UNet(n_input_channels=1, BATCH_SIZE=None, num_output_classes=2, pad='same', nonlinearity=lasagne.nonlinearities.elu, input_dim=(128, 128), base_n_filters=64, do_dropout=False):
    net = OrderedDict()
    net['input'] = InputLayer((BATCH_SIZE, n_input_channels, input_dim[0], input_dim[1]))

    net['contr_1_1'] = ConvLayer(net['input'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad)
    net['contr_1_2'] = ConvLayer(net['contr_1_1'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad)
    net['pool1'] = Pool2DLayer(net['contr_1_2'], 2)

    net['contr_2_1'] = ConvLayer(net['pool1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad)
    net['contr_2_2'] = ConvLayer(net['contr_2_1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad)
    net['pool2'] = Pool2DLayer(net['contr_2_2'], 2)

    net['contr_3_1'] = ConvLayer(net['pool2'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad)
    net['contr_3_2'] = ConvLayer(net['contr_3_1'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad)
    net['pool3'] = Pool2DLayer(net['contr_3_2'], 2)

    net['contr_4_1'] = ConvLayer(net['pool3'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad)
    net['contr_4_2'] = ConvLayer(net['contr_4_1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad)
    l = net['pool4'] = Pool2DLayer(net['contr_4_2'], 2)
    # the paper does not really describe where and how dropout is added. Feel free to try more options
    if do_dropout:
        l = DropoutLayer(l, p=0.4)

    net['encode_1'] = ConvLayer(l, base_n_filters*16, 3, nonlinearity=nonlinearity, pad=pad)
    net['encode_2'] = ConvLayer(net['encode_1'], base_n_filters*16, 3, nonlinearity=nonlinearity, pad=pad)
    net['upscale1'] = Upscale2DLayer(net['encode_2'], 2)

    net['concat1'] = ConcatLayer([net['upscale1'], net['contr_4_2']], cropping=(None, None, "center", "center"))
    net['expand_1_1'] = ConvLayer(net['concat1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad)
    net['expand_1_2'] = ConvLayer(net['expand_1_1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad)
    net['upscale2'] = Upscale2DLayer(net['expand_1_2'], 2)

    net['concat2'] = ConcatLayer([net['upscale2'], net['contr_3_2']], cropping=(None, None, "center", "center"))
    net['expand_2_1'] = ConvLayer(net['concat2'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad)
    net['expand_2_2'] = ConvLayer(net['expand_2_1'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad)
    net['upscale3'] = Upscale2DLayer(net['expand_2_2'], 2)

    net['concat3'] = ConcatLayer([net['upscale3'], net['contr_2_2']], cropping=(None, None, "center", "center"))
    net['expand_3_1'] = ConvLayer(net['concat3'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad)
    net['expand_3_2'] = ConvLayer(net['expand_3_1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad)
    net['upscale4'] = Upscale2DLayer(net['expand_3_2'], 2)

    net['concat4'] = ConcatLayer([net['upscale4'], net['contr_1_2']], cropping=(None, None, "center", "center"))
    net['expand_4_1'] = ConvLayer(net['concat4'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad)
    net['expand_4_2'] = ConvLayer(net['expand_4_1'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad)

    net['output_segmentation'] = ConvLayer(net['expand_4_2'], num_output_classes, 1, nonlinearity=None)
    net['dimshuffle'] = DimshuffleLayer(net['output_segmentation'], (1, 0, 2, 3))
    net['reshapeSeg'] = ReshapeLayer(net['dimshuffle'], (num_output_classes, -1))
    net['dimshuffle2'] = DimshuffleLayer(net['reshapeSeg'], (1, 0))
    net['output_flattened'] = NonlinearityLayer(net['dimshuffle2'], nonlinearity=lasagne.nonlinearities.softmax)

    return net


def build_UNet_relu(n_input_channels=1, BATCH_SIZE=None, num_output_classes=2, pad='same', input_dim=(128, 128), base_n_filters=64, do_dropout=False):
    nonlinearity = lasagne.nonlinearities.rectify
    net = OrderedDict()
    net['input'] = InputLayer((BATCH_SIZE, n_input_channels, input_dim[0], input_dim[1]))

    net['contr_1_1'] = ConvLayer(net['input'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu"))
    net['contr_1_2'] = ConvLayer(net['contr_1_1'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu"))
    net['pool1'] = Pool2DLayer(net['contr_1_2'], 2)

    net['contr_2_1'] = ConvLayer(net['pool1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu"))
    net['contr_2_2'] = ConvLayer(net['contr_2_1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu"))
    net['pool2'] = Pool2DLayer(net['contr_2_2'], 2)

    net['contr_3_1'] = ConvLayer(net['pool2'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu"))
    net['contr_3_2'] = ConvLayer(net['contr_3_1'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu"))
    net['pool3'] = Pool2DLayer(net['contr_3_2'], 2)

    net['contr_4_1'] = ConvLayer(net['pool3'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu"))
    net['contr_4_2'] = ConvLayer(net['contr_4_1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu"))
    l = net['pool4'] = Pool2DLayer(net['contr_4_2'], 2)
    # the paper does not really describe where and how dropout is added. Feel free to try more options
    if do_dropout:
        l = DropoutLayer(l, p=0.4)

    net['encode_1'] = ConvLayer(l, base_n_filters*16, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu"))
    net['encode_2'] = ConvLayer(net['encode_1'], base_n_filters*16, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu"))
    net['upscale1'] = Upscale2DLayer(net['encode_2'], 2)

    net['concat1'] = ConcatLayer([net['upscale1'], net['contr_4_2']], cropping=(None, None, "center", "center"))
    net['expand_1_1'] = ConvLayer(net['concat1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu"))
    net['expand_1_2'] = ConvLayer(net['expand_1_1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu"))
    net['upscale2'] = Upscale2DLayer(net['expand_1_2'], 2)

    net['concat2'] = ConcatLayer([net['upscale2'], net['contr_3_2']], cropping=(None, None, "center", "center"))
    net['expand_2_1'] = ConvLayer(net['concat2'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu"))
    net['expand_2_2'] = ConvLayer(net['expand_2_1'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu"))
    net['upscale3'] = Upscale2DLayer(net['expand_2_2'], 2)

    net['concat3'] = ConcatLayer([net['upscale3'], net['contr_2_2']], cropping=(None, None, "center", "center"))
    net['expand_3_1'] = ConvLayer(net['concat3'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu"))
    net['expand_3_2'] = ConvLayer(net['expand_3_1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu"))
    net['upscale4'] = Upscale2DLayer(net['expand_3_2'], 2)

    net['concat4'] = ConcatLayer([net['upscale4'], net['contr_1_2']], cropping=(None, None, "center", "center"))
    net['expand_4_1'] = ConvLayer(net['concat4'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu"))
    net['expand_4_2'] = ConvLayer(net['expand_4_1'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu"))

    net['output_segmentation'] = ConvLayer(net['expand_4_2'], num_output_classes, 1, nonlinearity=None)
    net['dimshuffle'] = DimshuffleLayer(net['output_segmentation'], (1, 0, 2, 3))
    net['reshapeSeg'] = ReshapeLayer(net['dimshuffle'], (num_output_classes, -1))
    net['dimshuffle2'] = DimshuffleLayer(net['reshapeSeg'], (1, 0))
    net['output_flattened'] = NonlinearityLayer(net['dimshuffle2'], nonlinearity=lasagne.nonlinearities.softmax)

    return net

def build_UNet_relu_BN(n_input_channels=1, BATCH_SIZE=None, num_output_classes=2, pad='same', input_dim=(128, 128), base_n_filters=64, do_dropout=False):
    nonlinearity = lasagne.nonlinearities.rectify
    net = OrderedDict()
    net['input'] = InputLayer((BATCH_SIZE, n_input_channels, input_dim[0], input_dim[1]))

    net['contr_1_1'] = batch_norm(ConvLayer(net['input'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu")))
    net['contr_1_2'] = batch_norm(ConvLayer(net['contr_1_1'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu")))
    net['pool1'] = Pool2DLayer(net['contr_1_2'], 2)

    net['contr_2_1'] = batch_norm(ConvLayer(net['pool1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu")))
    net['contr_2_2'] = batch_norm(ConvLayer(net['contr_2_1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu")))
    net['pool2'] = Pool2DLayer(net['contr_2_2'], 2)

    net['contr_3_1'] = batch_norm(ConvLayer(net['pool2'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu")))
    net['contr_3_2'] = batch_norm(ConvLayer(net['contr_3_1'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu")))
    net['pool3'] = Pool2DLayer(net['contr_3_2'], 2)

    net['contr_4_1'] = batch_norm(ConvLayer(net['pool3'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu")))
    net['contr_4_2'] = batch_norm(ConvLayer(net['contr_4_1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu")))
    l = net['pool4'] = Pool2DLayer(net['contr_4_2'], 2)
    # the paper does not really describe where and how dropout is added. Feel free to try more options
    if do_dropout:
        l = DropoutLayer(l, p=0.4)

    net['encode_1'] = batch_norm(ConvLayer(l, base_n_filters*16, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu")))
    net['encode_2'] = batch_norm(ConvLayer(net['encode_1'], base_n_filters*16, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu")))
    net['upscale1'] = Upscale2DLayer(net['encode_2'], 2)

    net['concat1'] = ConcatLayer([net['upscale1'], net['contr_4_2']], cropping=(None, None, "center", "center"))
    net['expand_1_1'] = batch_norm(ConvLayer(net['concat1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu")))
    net['expand_1_2'] = batch_norm(ConvLayer(net['expand_1_1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu")))
    net['upscale2'] = Upscale2DLayer(net['expand_1_2'], 2)

    net['concat2'] = ConcatLayer([net['upscale2'], net['contr_3_2']], cropping=(None, None, "center", "center"))
    net['expand_2_1'] = batch_norm(ConvLayer(net['concat2'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu")))
    net['expand_2_2'] = batch_norm(ConvLayer(net['expand_2_1'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu")))
    net['upscale3'] = Upscale2DLayer(net['expand_2_2'], 2)

    net['concat3'] = ConcatLayer([net['upscale3'], net['contr_2_2']], cropping=(None, None, "center", "center"))
    net['expand_3_1'] = batch_norm(ConvLayer(net['concat3'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu")))
    net['expand_3_2'] = batch_norm(ConvLayer(net['expand_3_1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu")))
    net['upscale4'] = Upscale2DLayer(net['expand_3_2'], 2)

    net['concat4'] = ConcatLayer([net['upscale4'], net['contr_1_2']], cropping=(None, None, "center", "center"))
    net['expand_4_1'] = batch_norm(ConvLayer(net['concat4'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu")))
    net['expand_4_2'] = batch_norm(ConvLayer(net['expand_4_1'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad, W=lasagne.init.HeNormal(gain="relu")))

    net['output_segmentation'] = ConvLayer(net['expand_4_2'], num_output_classes, 1, nonlinearity=None)
    net['dimshuffle'] = DimshuffleLayer(net['output_segmentation'], (1, 0, 2, 3))
    net['reshapeSeg'] = ReshapeLayer(net['dimshuffle'], (num_output_classes, -1))
    net['dimshuffle2'] = DimshuffleLayer(net['reshapeSeg'], (1, 0))
    net['output_flattened'] = NonlinearityLayer(net['dimshuffle2'], nonlinearity=lasagne.nonlinearities.softmax)

    return net