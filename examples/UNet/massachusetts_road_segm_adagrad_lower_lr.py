__author__ = 'Fabian Isensee'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import lasagne
import os
import sys
import fnmatch
import matplotlib.pyplot as plt
sys.path.append("../../modelzoo/")
from Unet import *
import theano.tensor as T
import theano
import cPickle
from time import sleep
from generators import batch_generator, threaded_generator, random_crop_generator, center_crop_generator, center_crop_seg_generator
from massachusetts_road_dataset_utils import prepare_dataset
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import DeepLearningBatchGeneratorUtils.SegmentationDataAugmentation as segDataAugm
import DeepLearningBatchGeneratorUtils.MultiThreadedGenerator as multiThreadedGen
import DeepLearningBatchGeneratorUtils.DataGeneratorBase
import os

class BatchGen(DeepLearningBatchGeneratorUtils.DataGeneratorBase.BatchGeneratorBase):
    def generate_train_batch(self):
        idx = np.random.choice(self._data[0].shape[0], self.BATCH_SIZE)
        return np.array(self._data[0][idx]), np.array(self._data[1][idx]), None

def plot_some_results(pred_fn, test_generator, n_images=10, prefix="road_segm"):
    fig_ctr = 0
    for data, seg in test_generator:
        res = pred_fn(data)
        for d, s, r in zip(data, seg, res):
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(d.transpose(1,2,0))
            plt.subplot(1, 3, 2)
            plt.imshow(s[0])
            plt.subplot(1, 3, 3)
            plt.imshow(r)
            plt.savefig("%s_%03.0f.png"%(prefix, fig_ctr))
            plt.close()
            fig_ctr += 1
            if fig_ctr > n_images:
                break
        if fig_ctr > n_images:
            break

def plot_learning_curve(train_accs, train_losses, val_accs, val_losses, fname):
    fig, ax1 = plt.subplots(figsize=(16, 12))
    n_epochs = len(train_accs)

    ax1.plot(range(n_epochs), train_losses, 'b--', linewidth=2)
    ax1.plot(range(n_epochs), val_losses, 'b', linewidth=2)
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')

    ax2 = ax1.twinx()
    ax2.plot(range(n_epochs), train_accs, 'r--', linewidth=2)
    ax2.plot(range(n_epochs), val_accs, color='r', linewidth=2)
    ax2.set_ylabel('accuracy')

    ax1.set_ylim((0, 1.))
    ax2.set_ylim((0.8,1.))
    plt.savefig(fname)
    plt.close()

def main():
    # only download dataset once. This takes a while.
    # heuristic that I included to make sure the dataset is only donwloaded and prepared once
    if not os.path.isfile("target_test.npy"):
        prepare_dataset()

    # set some hyper parameters. You should not have to touch anything if you have 4GB or more VRAM
    BATCH_SIZE = 20
    N_EPOCHS = 50
    N_BATCHES_PER_EPOCH = 200
    PATCH_SIZE = 512
    output_folder = "adagrad_lower_lr/"
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # load the prepared data. They have been converted to np arrays because they are much faster to load than single image files.
    # you will need some ram in order to have everything in memory.
    # If you are having RAM issues, change mmap_mode to 'r'. This will not load the entire array into memory but rather
    # read from disk the bits that we currently need
    # (if you have, copy your repository including the data to an SSD, otherwise it will take a long time to
    # generate batches)
    mmap_mode = 'r'
    data_train = np.load("data_train.npy", mmap_mode=mmap_mode)
    target_train = np.load("target_train.npy", mmap_mode=mmap_mode)
    data_valid = np.load("data_test.npy", mmap_mode=mmap_mode)
    target_valid = np.load("target_test.npy", mmap_mode=mmap_mode)
    data_test = np.load("data_valid.npy", mmap_mode=mmap_mode)
    target_test = np.load("target_valid.npy", mmap_mode=mmap_mode)

    # we are using pad='same' for simplicity (otherwise we would have to crop our ground truth).
    net = build_UNet_relu(n_input_channels=3, BATCH_SIZE=BATCH_SIZE, num_output_classes=2, pad='valid',
                     input_dim=(PATCH_SIZE, PATCH_SIZE),
                     base_n_filters=16, do_dropout=False)
    output_layer_for_loss = net["output_flattened"]

    # this is np.sum(target_train == 0) and np.sum(target_train == 1). No need to compute this every time
    class_frequencies = np.array([2374093357., 118906643.])
    # we are taking the log here because we want the net to focus more on the road pixels but not too much (otherwise
    # it would not be penalized enough for missclassifying terrain pixels which results in too many false positives)
    class_weights = class_frequencies[[1,0]]**1.0
    class_weights = class_weights / np.sum(class_weights) * 2.
    class_weights = class_weights.astype(np.float32)

    # if you wish to load pretrained weights you can uncomment this code and modify the file name
    # if you want, use my pretained weights:
    # val accuracy:  0.963513  val loss:  0.114994  val AUC score:  0.978996643458
    # https://www.dropbox.com/s/0vasqq491skf9iz/UNet_mass_road_segm_params.zip?dl=0
    '''with open("UNet_params_ep029.pkl", 'r') as f:
        params = cPickle.load(f)
        lasagne.layers.set_all_param_values(output_layer_for_loss, params)'''

    # some data augmentation. If you want better results you should invest more effort here. I left rotations and
    # deformations out for the sake of speed and simplicity
    train_generator_base = BatchGen((data_train, target_train), BATCH_SIZE)
    train_generator = segDataAugm.random_crop_generator(train_generator_base, (int(np.ceil(PATCH_SIZE * 2**0.5)), int(np.ceil(PATCH_SIZE * 2**0.5))))
    train_generator = segDataAugm.rotation_generator(train_generator)
    # train_generator = segDataAugm.segmentation_elastric_transform_generator(train_generator, 950, 40)
    train_generator = segDataAugm.center_crop_generator(train_generator, PATCH_SIZE)
    train_generator = segDataAugm.mirror_axis_generator(train_generator)
    train_generator = segDataAugm.center_crop_seg_generator(train_generator, 324)
    train_generator = multiThreadedGen.MultiThreadedGenerator(train_generator, 6, 10)
    train_generator._start()

    x_sym = T.tensor4()
    seg_sym = T.ivector()
    w_sym = T.vector()

    # add some weight decay
    l2_loss = lasagne.regularization.regularize_network_params(output_layer_for_loss, lasagne.regularization.l2) * 1e-4

    # the distinction between prediction_train and test is important only if we enable dropout
    prediction_train = lasagne.layers.get_output(output_layer_for_loss, x_sym, deterministic=False)
    # we could use a binary loss but I stuck with categorical crossentropy so that less code has to be changed if your
    # application has more than two classes
    loss = lasagne.objectives.categorical_crossentropy(prediction_train, seg_sym)
    loss *= w_sym
    loss = loss.mean()
    loss += l2_loss
    acc_train = T.mean(T.eq(T.argmax(prediction_train, axis=1), seg_sym), dtype=theano.config.floatX)

    prediction_test = lasagne.layers.get_output(output_layer_for_loss, x_sym, deterministic=True)
    loss_val = lasagne.objectives.categorical_crossentropy(prediction_test, seg_sym)

    # we multiply our loss by a weight map. In this example the weight map only increases the loss for road pixels and
    # decreases the loss for other pixels. We do this to ensure that the network puts more focus on getting the roads
    # right
    loss_val *= w_sym
    loss_val = loss_val.mean()
    loss_val += l2_loss
    acc = T.mean(T.eq(T.argmax(prediction_test, axis=1), seg_sym), dtype=theano.config.floatX)

    learning_rates = np.linspace(0.1, 0.0001, N_EPOCHS, dtype=np.float32)
    # momentums = np.linspace(0.9, 0.999, N_EPOCHS, dtype=np.float32)
    # learning rate has to be a shared variablebecause we decrease it with every epoch
    params = lasagne.layers.get_all_params(output_layer_for_loss, trainable=True)
    learning_rate = theano.shared(learning_rates[0])
    # momentum = theano.shared(momentums[0])
    # updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=momentum)
    updates = lasagne.updates.adagrad(loss, params, learning_rate=learning_rate)

    # create a convenience function to get the segmentation
    seg_output = lasagne.layers.get_output(net["output_segmentation"], x_sym, deterministic=True)
    seg_output = seg_output.argmax(1)

    train_fn = theano.function([x_sym, seg_sym, w_sym], [loss, acc_train], updates=updates)
    val_fn = theano.function([x_sym, seg_sym, w_sym], [loss_val, acc])
    get_segmentation = theano.function([x_sym], seg_output)
    # we need this for calculating the AUC score
    get_class_probas = theano.function([x_sym], prediction_test)

    # do the actual training
    all_training_losses = []
    all_validation_losses = []
    all_training_accs = []
    all_validation_accs = []
    val_loss_old = 999
    epoch = 0
    while epoch < N_EPOCHS:
        learning_rate.set_value(learning_rates[epoch])
        # momentum.set_value(momentums[epoch])
        print epoch
        losses_train = []
        n_batches = 0
        accuracies_train = []
        for data, target, _ in train_generator:
            # the output of the net has shape (BATCH_SIZE, N_CLASSES). We therefore need to flatten the segmentation so
            # that we can match it with the prediction via the crossentropy loss function
            loss, acc = train_fn(data.astype(np.float32), target.ravel(), class_weights[target.ravel()])
            losses_train.append(loss)
            accuracies_train.append(acc)
            n_batches += 1
            if n_batches > N_BATCHES_PER_EPOCH:
                break
        print "epoch: ", epoch, "\ntrain accuracy: ", np.mean(accuracies_train), " train loss: ", np.mean(losses_train)

        losses_val = []
        accuracies_val = []
        auc_val = []
        # there is no need for data augmentation on the validation. However we need patches of the same size which is why
        # we are using center crop generator
        # since the validation generator does not loop around we need to reinstantiate it for every epoch
        validation_generator = center_crop_generator(batch_generator(data_valid, target_valid, BATCH_SIZE, shuffle=False), PATCH_SIZE)
        validation_generator = center_crop_seg_generator(validation_generator, 324)
        validation_generator = threaded_generator(validation_generator, num_cached=10)
        for data, target in validation_generator:
            target_flat = target.ravel()
            loss, acc = val_fn(data.astype(np.float32), target_flat, class_weights[target_flat])
            losses_val.append(loss)
            accuracies_val.append(acc)
            auc_val.append(roc_auc_score(target_flat, get_class_probas(data)[:, 1]))
        print "val accuracy: ", np.mean(accuracies_val), " val loss: ", np.mean(losses_val), " val AUC score: ", np.mean(auc_val)
        if (epoch != 0) and (np.mean(losses_val) > 1.5 * val_loss_old):
            print "oops..."
            with open(output_folder + "UNet_params_ep%03.0f.pkl"%(epoch-1), 'r') as f:
                params = cPickle.load(f)
            with open(output_folder + "UNet_acc_losses_ep%03.0f.pkl"%(epoch-1), 'r') as f:
                [all_training_accs, all_training_losses, all_validation_accs, all_validation_losses] = cPickle.load(f)
            lasagne.layers.set_all_param_values(output_layer_for_loss, params)
        else:
            # save trained weights after each epoch
            with open(output_folder + "UNet_params_ep%03.0f.pkl"%epoch, 'w') as f:
                cPickle.dump(lasagne.layers.get_all_param_values(output_layer_for_loss), f)
            with open(output_folder + "UNet_acc_losses_ep%03.0f.pkl"%epoch, 'w') as f:
                cPickle.dump([all_training_accs, all_training_losses, all_validation_accs, all_validation_losses], f)
            # create some png files showing (raw image, ground truth, prediction). Of course we use the test set here ;-)
            test_gen = random_crop_generator(batch_generator(data_test, target_test, BATCH_SIZE), PATCH_SIZE)
            if not os.path.isdir(output_folder + "ep%02.0d/"%epoch):
                os.mkdir(output_folder + "ep%02.0d/"%epoch)
            plot_some_results(get_segmentation, test_gen, 30, output_folder + "ep%02.0d/"%epoch + "ep_%02.0d_road_segm"%epoch)
            epoch += 1
            all_training_accs.append(np.mean(accuracies_train))
            all_training_losses.append(np.mean(losses_train))
            all_validation_losses.append(np.mean(losses_val))
            all_validation_accs.append(np.mean(accuracies_val))
            plot_learning_curve(all_training_accs, all_training_losses, all_validation_accs, all_validation_losses, output_folder + "road_segm_learn_curve.png")
            val_loss_old = np.mean(losses_val)
    train_generator._finish()

if __name__ == "__main__":
    main()
