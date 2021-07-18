import argparse
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
from fid_score import evaluate_fid_score

# SEED = 1237
SEED = None
np.random.seed(SEED)
DPI = None
HEADER = '\033[95m'
OK_BLUE = '\033[94m'
OK_GREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
END_C = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'


def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return


def load_train_dataset(name, root_folder='./'):
    if name.lower() == 'mnist':
        (x, _), (_, _) = tf.keras.datasets.mnist.load_data()
        side_length = 28
        channels = 1
    elif name.lower() == 'fashion':
        (temp, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
        x = deepcopy(temp)
        side_length = 28
        channels = 1
    elif name.lower() == 'cifar10':
        (x, _), (_, _) = tf.keras.datasets.cifar10.load_data()
        side_length = 32
        channels = 3
    elif name.lower() == 'celeba140':
        data_folder = os.path.join(root_folder, 'data', name.lower())
        x = np.load(os.path.join(data_folder, 'train.npy'))
        side_length = 64
        channels = 3
    elif name.lower() == 'celeba':
        data_folder = os.path.join(root_folder, 'data', name)
        x = np.load(os.path.join(data_folder, 'train.npy'))
        side_length = 64
        channels = 3
    else:
        raise Exception('No such dataset called {}.'.format(name))
    np.random.shuffle(x)
    return x.reshape([-1, side_length, side_length, channels]), side_length, channels


def load_test_dataset(name, root_folder='./'):
    if name.lower() == 'mnist':
        (_, _), (x, _) = tf.keras.datasets.mnist.load_data()
        side_length = 28
        channels = 1
    elif name.lower() == 'fashion':
        (_, _), (temp, _) = tf.keras.datasets.fashion_mnist.load_data()
        x = deepcopy(temp)
        side_length = 28
        channels = 1
    elif name.lower() == 'cifar10':
        (_, _), (x, _) = tf.keras.datasets.cifar10.load_data()
        side_length = 32
        channels = 3
    elif name.lower() == 'celeba140':
        data_folder = os.path.join(root_folder, 'data', name.lower())
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 64
        channels = 3
    elif name.lower() == 'celeba':
        data_folder = os.path.join(root_folder, 'data', name)
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 64
        channels = 3
    else:
        raise Exception('No such dataset called {}.'.format(name))
    np.random.shuffle(x)
    return x.reshape([-1, side_length, side_length, channels]), side_length, channels


class TrainingDataGenerator:
    def __init__(self, x_train, side_length, channels):
        self.x_train = x_train / 255.0
        self.n_digits = self.x_train.shape[0]
        self.side_length = side_length
        self.channels = channels

    def get_batch(self, bs):
        image_indices = np.random.randint(0, self.n_digits, bs)
        return self.x_train[image_indices, :, :, :]


def image_grid(images, sv_path, dataset):
    size = int(images.shape[0] ** 0.5)
    images = (images - np.min(images)) / (np.max(images) - np.min(images))
    plt.figure(figsize=(20, 20))
    for i in range(images.shape[0]):
        plt.subplot(size, size, i + 1)
        image = images[i, :, :, :]
        if dataset == 'mnist' or dataset == 'fashion':
            image = np.reshape(image, [28, 28])
            plt.imshow(image, cmap='gray')
        elif dataset == 'celeba' or dataset == 'celeba140' or dataset == 'cifar10':
            plt.imshow(image)
        plt.axis('off')
    plt.savefig(sv_path, dpi=DPI)
    plt.close('all')
    return


def plot_graph(x, y, x_label, y_label, samples_dir, img_name, z_dim, noise_dim, n_col=1):
    plt.close('all')
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(b=True, which='both')
    plt.annotate(f'(Z_DIM: {z_dim}, N_DIM: {noise_dim})', xy=(0.3, 0.5), xycoords='axes fraction')
    plt.savefig(samples_dir + img_name, dpi=DPI)


def sample_z(batch, z_dim, sampler='one_hot', num_class=10, label_index=None):
    if sampler == 'one_hot':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return np.hstack((np.random.randn(batch, z_dim), np.eye(num_class)[label_index]))
    elif sampler == 'uniform':
        return np.random.uniform(-1., 1., size=[batch, z_dim])
    elif sampler == 'normal':
        return np.random.randn(batch, z_dim)
    elif sampler == 'mix_gauss':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return np.random.randn(batch, z_dim) + np.eye(num_class, z_dim)[label_index]


class FlexAESR:
    def __init__(self, dataset, bs, z_dim, gan_lat_sampler, gan_lat_dim, num_class, lr, models_dir, samples_dir,
                 gen_dir, recon_dir, training_steps, save_interval, plot_interval, bn_axis, gradient_penalty_weight,
                 disc_training_ratio, ae_loss_fn, gan_loss_fn, gp, sess):
        self.dataset = dataset
        self.bs = bs
        self.z_dim = z_dim
        self.gan_lat_sampler = gan_lat_sampler
        self.gan_lat_dim = gan_lat_dim
        self.num_class = num_class
        self.noise_dim = sample_z(1, self.gan_lat_dim, self.gan_lat_sampler, self.num_class).shape[1]
        self.lr = lr
        self.model_dir = models_dir
        self.samples_dir = samples_dir
        self.gen_dir = gen_dir
        self.recon_dir = recon_dir
        self.training_steps = training_steps
        self.save_interval = save_interval
        self.plot_interval = plot_interval
        self.bn_axis = bn_axis
        self.gradient_penalty_weight = gradient_penalty_weight
        self.disc_training_ratio = disc_training_ratio
        self.ae_loss_fn = ae_loss_fn
        self.gan_loss_fn = gan_loss_fn
        self.GP = gp
        self.is_training = tf.compat.v1.placeholder_with_default(False, (), 'is_training')
        self.sess = sess
        self.kernel_initializer = tf.compat.v1.glorot_normal_initializer()
        self.bias_initializer = None
        self.enc_kernel_reg = None
        self.dec_kernel_reg = None
        self.enc_kernel_constraint = None
        self.dec_kernel_constraint = None
        self.disc_kernel_constraint = None
        self.gen_kernel_constraint = None
        self.lat_reg = None
        self.x_train, self.side_length, self.channels = load_train_dataset(name=dataset)
        self._build_model()
        self._loss()
        self._trainer()

    def _build_encoder(self, inp):
        encoded = inp
        encoded = tf.compat.v1.layers.conv2d(inputs=encoded, filters=64, kernel_size=(4, 4), strides=(2, 2),
                                             padding='same', activation=None, name='enc-conv-0')
        encoded = tf.nn.relu(encoded)
        encoded = tf.compat.v1.layers.conv2d(inputs=encoded, filters=128, kernel_size=(4, 4), strides=(2, 2),
                                             padding='same', activation=None, name='enc-conv-1')
        encoded = tf.layers.batch_normalization(inputs=encoded, axis=self.bn_axis,
                                                training=self.is_training, name='enc-bn-1')
        encoded = tf.nn.relu(encoded)

        encoded = tf.compat.v1.layers.flatten(inputs=encoded, name='enc-flatten')
        encoded = tf.compat.v1.layers.dense(inputs=encoded, units=1024, activation=None, name='enc-dense-0')
        encoded = tf.layers.batch_normalization(inputs=encoded, training=self.is_training,
                                                name='enc-bn-2')
        encoded = tf.nn.relu(encoded)
        encoded = tf.compat.v1.layers.dense(inputs=encoded, units=self.z_dim, activation=None, name='enc-final')

        return encoded

    def _build_decoder(self, inp):
        decoded = inp
        decoded = tf.compat.v1.layers.dense(inputs=decoded, units=1024, activation=None,
                                            kernel_initializer=self.kernel_initializer,
                                            bias_initializer=self.bias_initializer, name='dec-dense-0')
        decoded = tf.layers.batch_normalization(inputs=decoded, training=self.is_training,
                                                name='dec-bn-0')
        decoded = tf.nn.relu(decoded)
        decoded = tf.compat.v1.layers.dense(inputs=decoded,
                                            units=np.prod((128, self.side_length // 4, self.side_length // 4)),
                                            activation=None, kernel_initializer=self.kernel_initializer,
                                            bias_initializer=self.bias_initializer, name='dec-dense-1')
        decoded = tf.layers.batch_normalization(inputs=decoded, training=self.is_training,
                                                name='dec-bn-1')
        decoded = tf.nn.relu(decoded)
        decoded = tf.compat.v1.reshape(tensor=decoded, shape=(-1, self.side_length // 4, self.side_length // 4, 128),
                                       name='dec-reshape')

        decoded = tf.compat.v1.layers.conv2d_transpose(inputs=decoded, filters=128, kernel_size=(4, 4),
                                                       strides=(2, 2), padding='same', activation=None,
                                                       kernel_initializer=self.kernel_initializer,
                                                       bias_initializer=self.bias_initializer,
                                                       name='dec-trans-conv-0')
        decoded = tf.layers.batch_normalization(inputs=decoded, axis=self.bn_axis,
                                                training=self.is_training, name='dec-bn-2')
        decoded = tf.nn.relu(decoded)

        decoded = tf.compat.v1.layers.conv2d_transpose(inputs=decoded, filters=64, kernel_size=(4, 4),
                                                       strides=(2, 2), padding='same', activation=None,
                                                       kernel_initializer=self.kernel_initializer,
                                                       bias_initializer=self.bias_initializer,
                                                       name='dec-trans-conv-1')
        decoded = tf.layers.batch_normalization(inputs=decoded, axis=self.bn_axis,
                                                training=self.is_training, name='dec-bn-3')
        decoded = tf.nn.relu(decoded)

        decoded = tf.compat.v1.layers.conv2d(inputs=decoded, filters=self.channels, kernel_size=(3, 3), strides=(1, 1),
                                             padding='same', activation=tf.nn.sigmoid, name='dec-final')
        return decoded

    def _build_generator(self, inp):
        gen_op = inp
        n_neurons = 1024
        for i in range(4):
            gen_op = tf.compat.v1.layers.dense(inputs=gen_op, units=n_neurons, activation=None,
                                               kernel_initializer=self.kernel_initializer,
                                               kernel_constraint=self.gen_kernel_constraint,
                                               name=f'gen-dense-{i}')
            gen_op = tf.nn.relu(gen_op)

        gen_op = tf.compat.v1.layers.dense(inputs=gen_op, units=self.z_dim, activation=None,
                                           kernel_initializer=self.kernel_initializer,
                                           kernel_constraint=self.gen_kernel_constraint,
                                           activity_regularizer=self.lat_reg, name='gen-final')

        return gen_op

    def _build_discriminator(self, inp):
        decision = inp
        n_neurons = 1024
        for i in range(4):
            decision = tf.compat.v1.layers.dense(inputs=decision, units=n_neurons, activation=tf.nn.relu,
                                                 kernel_initializer=self.kernel_initializer,
                                                 kernel_constraint=self.disc_kernel_constraint, name=f'disc-dense-{i}')

        decision = tf.compat.v1.layers.dense(inputs=decision, units=1, activation=None,
                                             kernel_initializer=self.kernel_initializer,
                                             kernel_constraint=self.disc_kernel_constraint, name='disc-final')

        return decision

    def _build_img_discriminator(self, inp):
        logit = tf.compat.v1.layers.conv2d(inputs=inp, filters=64, kernel_size=(4, 4), strides=(2, 2),
                                           padding='same', activation=tf.nn.relu, name='img-disc-conv-0')
        logit = tf.compat.v1.layers.conv2d(inputs=logit, filters=128, kernel_size=(4, 4), strides=(2, 2),
                                           padding='same', activation=tf.nn.relu, name='img-disc-conv-1')
        logit = tf.compat.v1.layers.flatten(inputs=logit, name='img-disc-flatten')
        logit = tf.compat.v1.layers.dense(inputs=logit, units=1024, activation=tf.nn.relu, name='img-disc-dense-0')
        logit = tf.compat.v1.layers.dense(inputs=logit, units=1, activation=None, name='img-disc-final')

        return logit

    def _build_model(self):
        self.inp_img = tf.compat.v1.placeholder(tf.float32,
                                                [None, self.side_length, self.side_length, self.channels],
                                                name='input-image')
        self.inp_img_1 = tf.compat.v1.placeholder(tf.float32,
                                                  [None, self.side_length, self.side_length, self.channels],
                                                  name='input-image-1')
        self.prior_sample = tf.compat.v1.placeholder(tf.float32, [None, self.noise_dim], name='prior-sample')
        self.bs_ph = tf.compat.v1.placeholder(tf.int32, [1], name='batch-size-ph')
        with tf.compat.v1.variable_scope('flexaesr'):
            self.enc_lat = self._build_encoder(self.inp_img)
            self.reconstructed_img = self._build_decoder(self.enc_lat)
            self.gen_lat = self._build_generator(self.prior_sample)
            self.disc_op_for_enc_lat = self._build_discriminator(self.enc_lat)
            self.real_img_logit = self._build_img_discriminator(self.inp_img)

        with tf.compat.v1.variable_scope('flexaesr', reuse=True):
            self.disc_op_for_gen_lat = self._build_discriminator(self.gen_lat)
            self.generated_img = self._build_decoder(self.gen_lat)
            # For Latent Traversal
            alpha = tf.random.uniform(
                shape=[self.bs_ph[0], 1],
                minval=0.,
                maxval=1.,
                seed=SEED
            )
            self.enc_lat_1 = self._build_encoder(self.inp_img_1)
            self.latent_interpolates = alpha * self.enc_lat + (1 - alpha) * self.enc_lat_1

            self.latent_interpolated_images = self._build_decoder(self.latent_interpolates)
            self.fake_img_logit = self._build_img_discriminator(self.latent_interpolated_images)

        if self.GP:
            beta = tf.random.uniform(
                shape=[self.bs_ph[0], 1],
                minval=0.,
                maxval=1.,
                seed=SEED
            )
            gamma = tf.random.uniform(
                shape=[self.bs_ph[0], 1, 1, 1],
                minval=0.,
                maxval=1.,
                seed=SEED
            )
            lat_differences = self.enc_lat - self.gen_lat
            lat_interpolates = self.gen_lat + (beta * lat_differences)
            img_differences = self.inp_img - self.latent_interpolated_images
            img_interpolates = self.latent_interpolated_images + (gamma * img_differences)
            with tf.compat.v1.variable_scope('flexaesr', reuse=True):
                lat_gradients = tf.gradients(self._build_discriminator(lat_interpolates), [lat_interpolates])[0]
                img_gradients = tf.gradients(self._build_img_discriminator(img_interpolates), [img_interpolates])[0]
            self.lat_slopes = tf.sqrt(tf.reduce_sum(tf.square(lat_gradients), reduction_indices=[1]))
            self.img_slopes = tf.sqrt(tf.reduce_sum(tf.square(img_gradients), reduction_indices=[1, 2, 3]))

        return

    def _loss(self):
        # Reconstruction Loss
        if self.ae_loss_fn == 'mse':
            self.ae_loss = self.side_length * self.side_length * self.channels * tf.keras.losses.MeanSquaredError()(
                self.inp_img, self.reconstructed_img)
        elif self.ae_loss_fn == 'mae':
            self.ae_loss = tf.keras.losses.MeanAbsoluteError()(self.inp_img, self.reconstructed_img)
        elif self.ae_loss_fn == 'bce':
            self.ae_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(self.inp_img,
                                                                                 self.reconstructed_img)
        else:
            raise Exception(f'Auto-encoder loss: {self.ae_loss_fn} is not implemented.')

        # GAN Loss

        if self.gan_loss_fn == 'wgan':
            self.disc_loss = tf.math.reduce_mean(self.disc_op_for_gen_lat) - tf.math.reduce_mean(
                self.disc_op_for_enc_lat)
            self.gen_loss = -tf.math.reduce_mean(self.disc_op_for_gen_lat)
            self.enc_loss = tf.math.reduce_mean(self.disc_op_for_enc_lat)

            self.img_disc_loss = tf.math.reduce_mean(self.fake_img_logit) - tf.math.reduce_mean(self.real_img_logit)
            self.ae_smoothing_loss = -tf.math.reduce_mean(self.fake_img_logit)
        else:
            raise Exception(f'GAN loss: {self.gan_loss_fn} is not implemented.')

        if self.GP:
            lat_gradient_penalty = tf.math.reduce_mean((self.lat_slopes - 1.) ** 2)
            self.disc_loss += self.gradient_penalty_weight * lat_gradient_penalty

            img_gradient_penalty = tf.math.reduce_mean((self.img_slopes - 1.) ** 2)
            self.img_disc_loss += self.gradient_penalty_weight * img_gradient_penalty

        return

    def _trainer(self):
        self.ae_lr_decay = tf.compat.v1.placeholder(tf.float32, [1], name='ae-lr-decay')
        self.gen_lr_decay = tf.compat.v1.placeholder(tf.float32, [1], name='gen-lr-decay')
        self.disc_lr_decay = tf.compat.v1.placeholder(tf.float32, [1], name='disc-lr-decay')
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

        self.ae_variables = [var for var in tf.compat.v1.trainable_variables() if
                             (var.name.startswith('flexaesr/enc-') or
                              var.name.startswith('flexaesr/dec-'))]
        self.enc_variables = [var for var in tf.compat.v1.trainable_variables() if
                              var.name.startswith('flexaesr/enc-')]
        self.dec_variables = [var for var in tf.compat.v1.trainable_variables() if
                              var.name.startswith('flexaesr/dec-')]
        self.gen_variables = [var for var in tf.compat.v1.trainable_variables() if
                              var.name.startswith('flexaesr/gen-')]
        self.disc_variables = [var for var in tf.compat.v1.trainable_variables() if
                               var.name.startswith('flexaesr/disc-')]
        self.img_disc_variables = [var for var in tf.compat.v1.trainable_variables() if
                                   var.name.startswith('flexaesr/img-disc-')]

        with tf.control_dependencies(update_ops):
            self.ae_trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=2 * self.lr, name='ae-trainer') \
                .minimize(self.ae_loss * self.ae_lr_decay, var_list=self.ae_variables)
            self.enc_trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr / 10, beta1=0.0, beta2=0.9,
                                                                epsilon=1e-08, name='enc-trainer') \
                .minimize(self.enc_loss * self.ae_lr_decay, var_list=self.enc_variables)
            self.ae_trainer2 = tf.compat.v1.train.AdamOptimizer(learning_rate=2 * self.lr, beta1=0.5, beta2=0.9,
                                                                epsilon=1e-08, name='ae-trainer-2') \
                .minimize(self.ae_smoothing_loss * self.ae_lr_decay, var_list=self.ae_variables)
            self.gen_trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=2 * self.lr, beta1=0.0, beta2=0.9,
                                                                epsilon=1e-08, name='gen-trainer') \
                .minimize(self.gen_loss * self.gen_lr_decay, var_list=self.gen_variables)
            self.disc_trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr, beta1=0.0, beta2=0.9,
                                                                 epsilon=1e-08, name='disc-trainer') \
                .minimize(self.disc_loss * self.disc_lr_decay, var_list=self.disc_variables)
            self.img_disc_trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=2 * self.lr, beta1=0.5, beta2=0.9,
                                                                     epsilon=1e-08, name='img-disc-trainer') \
                .minimize(self.img_disc_loss * self.disc_lr_decay, var_list=self.img_disc_variables)

        return

    def train(self):
        sess = self.sess
        saver = tf.compat.v1.train.Saver(max_to_keep=(self.training_steps // self.save_interval))
        sess.run(tf.compat.v1.global_variables_initializer())

        ae_loss_buf = []
        enc_loss_buf = []
        gen_loss_buf = []
        disc_loss_buf = []
        ae_smoothing_loss_buf = []
        img_disc_loss_buf = []
        steps_buf = []
        ae_lr_decay_val = np.asarray([1])
        gen_lr_decay_val = np.asarray([1])
        disc_lr_decay_val = np.asarray([1])
        true_data_gen = TrainingDataGenerator(self.x_train, self.side_length, self.channels)

        for step in range(self.training_steps):
            image_batch = true_data_gen.get_batch(bs=self.bs)
            image_batch_1 = true_data_gen.get_batch(bs=self.bs)
            gaussian_sample = sample_z(self.bs, self.gan_lat_dim, self.gan_lat_sampler, self.num_class)
            ae_feed_dict = {self.inp_img: image_batch, self.inp_img_1: image_batch_1,
                            self.prior_sample: gaussian_sample, self.bs_ph: np.asarray([self.bs]),
                            self.ae_lr_decay: ae_lr_decay_val, self.gen_lr_decay: gen_lr_decay_val,
                            self.disc_lr_decay: disc_lr_decay_val, self.is_training: True}
            sess.run(self.ae_trainer, feed_dict=ae_feed_dict)

            for _ in range(self.disc_training_ratio):
                image_batch = true_data_gen.get_batch(bs=self.bs)
                image_batch_1 = true_data_gen.get_batch(bs=self.bs)
                gaussian_sample = sample_z(self.bs, self.gan_lat_dim, self.gan_lat_sampler, self.num_class)
                disc_feed_dict = {self.inp_img: image_batch, self.inp_img_1: image_batch_1,
                                  self.prior_sample: gaussian_sample, self.bs_ph: np.asarray([self.bs]),
                                  self.ae_lr_decay: ae_lr_decay_val, self.gen_lr_decay: gen_lr_decay_val,
                                  self.disc_lr_decay: disc_lr_decay_val, self.is_training: True}
                sess.run(self.disc_trainer, feed_dict=disc_feed_dict)
                sess.run(self.img_disc_trainer, feed_dict=disc_feed_dict)

            gen_feed_dict = {self.inp_img: image_batch, self.inp_img_1: image_batch_1,
                             self.prior_sample: gaussian_sample, self.bs_ph: np.asarray([self.bs]),
                             self.ae_lr_decay: ae_lr_decay_val, self.gen_lr_decay: gen_lr_decay_val,
                             self.disc_lr_decay: disc_lr_decay_val, self.is_training: True}
            sess.run(self.ae_trainer2, feed_dict=gen_feed_dict)
            sess.run(self.gen_trainer, feed_dict=gen_feed_dict)
            sess.run(self.enc_trainer, feed_dict=gen_feed_dict)

            if step % (self.plot_interval // 10) == 0:
                image_batch = true_data_gen.get_batch(bs=100)
                image_batch_1 = true_data_gen.get_batch(bs=100)
                gaussian_sample = sample_z(100, self.gan_lat_dim, self.gan_lat_sampler, self.num_class)
                feed_dict = {self.inp_img: image_batch, self.inp_img_1: image_batch_1,
                             self.prior_sample: gaussian_sample, self.bs_ph: np.asarray([100]),
                             self.ae_lr_decay: ae_lr_decay_val, self.gen_lr_decay: gen_lr_decay_val,
                             self.disc_lr_decay: disc_lr_decay_val, self.is_training: False}
                reconstructed_images = sess.run(self.reconstructed_img, feed_dict=feed_dict)
                image_grid(images=image_batch, sv_path='{}true_iter_{}.png'.format(self.samples_dir, step),
                           dataset=self.dataset)
                image_grid(images=reconstructed_images, sv_path='{}recon_iter_{}.png'.format(self.samples_dir, step),
                           dataset=self.dataset)
                generated_images = sess.run(self.generated_img, feed_dict=feed_dict)
                image_grid(images=generated_images, sv_path='{}generated_iter_{}.png'.format(self.samples_dir, step),
                           dataset=self.dataset)
                interpolated_images = sess.run(self.latent_interpolated_images, feed_dict=feed_dict)
                image_grid(images=interpolated_images, sv_path='{}interpolated_iter_{}.png'.format(self.samples_dir,
                                                                                                   step),
                           dataset=self.dataset)
                ae_loss_val = sess.run(self.ae_loss, feed_dict=feed_dict)
                enc_loss_val = sess.run(self.enc_loss, feed_dict=feed_dict)
                gen_loss_val = sess.run(self.gen_loss, feed_dict=feed_dict)
                disc_loss_val = sess.run(self.disc_loss, feed_dict=feed_dict)
                ae_smoothing_loss_val = sess.run(self.ae_smoothing_loss, feed_dict=feed_dict)
                img_disc_loss_val = sess.run(self.img_disc_loss, feed_dict=feed_dict)
                ae_loss_buf.append(ae_loss_val)
                enc_loss_buf.append(enc_loss_val)
                gen_loss_buf.append(gen_loss_val)
                disc_loss_buf.append(disc_loss_val)
                ae_smoothing_loss_buf.append(ae_smoothing_loss_val)
                img_disc_loss_buf.append(img_disc_loss_val)
                steps_buf.append(step)

                print(HEADER + f'Training an FlexAE-SR Model on {self.dataset.upper()}\n' + END_C)
                print(HEADER + f'Latent Dim: {self.z_dim}; Noise Dim: {self.noise_dim}; AE Loss Fn: {self.ae_loss_fn}, '
                               f'GAN Loss Fn: {self.gan_loss_fn}; GP Status: {self.GP}\n' + END_C)
                print(WARNING + f'Step: {steps_buf[-1]}, ' + END_C + OK_BLUE +
                      f'AE Loss: {ae_loss_buf[-1]}, Disc Loss: {disc_loss_buf[-1]}, '
                      f'Enc Loss: {enc_loss_buf[-1]}, '
                      f'Gen Loss: {gen_loss_buf[-1]}, '
                      f'Image Disc Loss: {img_disc_loss_buf[-1]}, '
                      f'AE Smoothing Loss: {ae_smoothing_loss_buf[-1]}\n'
                      + END_C)

            if step % self.plot_interval == 0 and step > 0:
                plot_graph(x=steps_buf, y=ae_loss_buf, x_label='Steps', y_label='AE Loss', samples_dir=self.samples_dir,
                           img_name='ae_loss.png', z_dim=self.z_dim, noise_dim=self.noise_dim)
                plot_graph(x=steps_buf, y=enc_loss_buf, x_label='Steps', y_label='Enc Loss',
                           samples_dir=self.samples_dir, img_name='enc_loss.png', z_dim=self.z_dim,
                           noise_dim=self.noise_dim)
                plot_graph(x=steps_buf, y=gen_loss_buf, x_label='Steps', y_label='Generator Loss',
                           samples_dir=self.samples_dir, img_name='gen_loss.png', z_dim=self.z_dim,
                           noise_dim=self.noise_dim)
                plot_graph(x=steps_buf, y=disc_loss_buf, x_label='Steps', y_label='Discriminator Loss',
                           samples_dir=self.samples_dir, img_name='disc_loss.png', z_dim=self.z_dim,
                           noise_dim=self.noise_dim)
                plot_graph(x=steps_buf, y=img_disc_loss_buf, x_label='Steps', y_label='Image Discriminator Loss',
                           samples_dir=self.samples_dir, img_name='img_disc_loss.png', z_dim=self.z_dim,
                           noise_dim=self.noise_dim)
                plot_graph(x=steps_buf, y=ae_smoothing_loss_buf, x_label='Steps', y_label='AE Smoothing Loss',
                           samples_dir=self.samples_dir, img_name='ae_smoothing_loss.png', z_dim=self.z_dim,
                           noise_dim=self.noise_dim)

            if step % self.save_interval == 0 and step > 0:
                saver.save(sess, os.path.join(self.model_dir, 'FlexAESR.model'), global_step=step)
                gen_img = np.ndarray(shape=(10000, self.side_length, self.side_length, self.channels))

                for j in range(10000 // 100):
                    z_sample = sample_z(100, self.gan_lat_dim, self.gan_lat_sampler, self.num_class)
                    feed_dict = {self.bs_ph: np.asarray([100]), self.prior_sample: z_sample, self.is_training: False}
                    gen_img_batch = sess.run(self.generated_img, feed_dict=feed_dict)
                    gen_img[(j * 100):((j + 1) * 100)] = gen_img_batch.reshape(-1, self.side_length,
                                                                               self.side_length, self.channels)

                np.save('{}generated_{}_images_{}.npy'.format(self.gen_dir, self.dataset, step), gen_img)

                test_x, _, _ = load_test_dataset(self.dataset, './')
                recon_img = np.ndarray(shape=(10000, self.side_length, self.side_length, self.channels))
                for j in range(10000 // 100):
                    feed_dict = {self.bs_ph: np.asarray([100]),
                                 self.inp_img: test_x[(j * 100):((j + 1) * 100)] / np.float32(255.0),
                                 self.is_training: False}
                    recon_img[(j * 100):((j + 1) * 100)] = sess.run(
                        self.reconstructed_img, feed_dict=feed_dict).reshape(-1, self.side_length, self.side_length,
                                                                             self.channels)

                np.save('{}reconstructed_{}_images_{}.npy'.format(self.recon_dir, self.dataset, step), recon_img)

        with open(self.samples_dir + 'ae_loss.txt', 'wb') as fp:
            pickle.dump(ae_loss_buf, fp)
        with open(self.samples_dir + 'enc_loss.txt', 'wb') as fp:
            pickle.dump(enc_loss_buf, fp)
        with open(self.samples_dir + 'gen_loss.txt', 'wb') as fp:
            pickle.dump(gen_loss_buf, fp)
        with open(self.samples_dir + 'disc_loss.txt', 'wb') as fp:
            pickle.dump(disc_loss_buf, fp)
        with open(self.samples_dir + 'ae_smoothing_loss.txt', 'wb') as fp:
            pickle.dump(ae_smoothing_loss_buf, fp)
        with open(self.samples_dir + 'img_disc_loss.txt', 'wb') as fp:
            pickle.dump(img_disc_loss_buf, fp)
        with open(self.samples_dir + 'plot_steps.txt', 'wb') as fp:
            pickle.dump(steps_buf, fp)

        return


def evaluate_fid(image_dir, prefix, dataset, step, root_folder='./'):
    tf.compat.v1.reset_default_graph()
    fake_images = np.load('{}{}_{}_images_{}.npy'.format(image_dir, prefix, dataset, step))
    try:
        fid_val = evaluate_fid_score(fake_images, dataset, root_folder, True)
    except:
        fid_val = np.inf
    return fid_val


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['mnist', 'fashion', 'cifar10', 'celeba'])
parser.add_argument('--z_dim', type=int, default=-1)
parser.add_argument('--gan_lat_sampler', type=str, default='normal',
                    choices=['one_hot', 'uniform', 'normal', 'mix_gauss'])
parser.add_argument('--gan_lat_dim', type=int, default=-1)
parser.add_argument('--num_class', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--save_interval', type=int, default=1000)
parser.add_argument('--plot_interval', type=int, default=1000)
parser.add_argument('--training_steps', type=int, default=130001)
parser.add_argument('--bn_axis', type=int, default=-1)
parser.add_argument('--ae_loss_fn', type=str, default='mse', choices=['mse', 'mae', 'bce'])
parser.add_argument('--gan_loss_fn', type=str, default='wgan',
                    choices=['wgan'])
parser.add_argument('--gp', type=int, default=1, choices=[0, 1])
parser.add_argument('--gradient_penalty_weight', type=float, default=10)
parser.add_argument('--disc_training_ratio', type=int, default=5)

args = parser.parse_args()

if args.z_dim == -1:
    if args.dataset == 'mnist' or args.dataset == 'fashion':
        args.z_dim = 32
    elif args.dataset == 'cifar10':
        args.z_dim = 128
    elif args.dataset == 'celeba':
        args.z_dim = 128
    else:
        raise Exception(f'Please Provide Latent Dimension of AE for {args.dataset} Dataset.')

if args.gan_lat_dim == -1:
    args.gan_lat_dim = args.z_dim

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

trained_models_dir = f'./FlexAESR_Models/{args.dataset}/zdim{args.z_dim}_ndim{args.gan_lat_dim}/'
training_data_dir = f'./FlexAESR_Samples/{args.dataset}/zdim{args.z_dim}_ndim{args.gan_lat_dim}/'
reconstructed_data_dir = f'./FlexAESR_Reconstructed/{args.dataset}/zdim{args.z_dim}_ndim{args.gan_lat_dim}/'
generated_data_dir = f'./FlexAESR_Generated/{args.dataset}/zdim{args.z_dim}_ndim{args.gan_lat_dim}/'
create_directory(trained_models_dir)
create_directory(training_data_dir)
create_directory(reconstructed_data_dir)
create_directory(generated_data_dir)

tf.compat.v1.reset_default_graph()
model = FlexAESR(dataset=args.dataset, bs=args.batch_size, z_dim=args.z_dim, gan_lat_sampler=args.gan_lat_sampler,
                 gan_lat_dim=args.gan_lat_dim, num_class=args.num_class, lr=args.lr, models_dir=trained_models_dir,
                 samples_dir=training_data_dir, gen_dir=generated_data_dir, recon_dir=reconstructed_data_dir,
                 training_steps=args.training_steps, save_interval=args.save_interval, plot_interval=args.plot_interval,
                 bn_axis=args.bn_axis, gradient_penalty_weight=args.gradient_penalty_weight,
                 disc_training_ratio=args.disc_training_ratio, ae_loss_fn=args.ae_loss_fn, gan_loss_fn=args.gan_loss_fn,
                 gp=args.gp, sess=tf.compat.v1.Session(config=config))
model.train()

print(HEADER + 'Training Complete' + END_C)

print(HEADER + 'Evaluating Generation FID Score' + END_C)

fid_buf = []
step_buf = []

for i in range(80 * args.save_interval, args.training_steps, args.save_interval):
    tf.compat.v1.reset_default_graph()
    fid_val = evaluate_fid(image_dir=generated_data_dir, prefix='generated', dataset=args.dataset, step=i,
                           root_folder='./')
    print(i, fid_val)
    fid_buf.append(fid_val)
    step_buf.append(i)
with open(generated_data_dir + 'fid_buf.txt', 'wb') as f:
    pickle.dump(fid_buf, f)
with open(generated_data_dir + 'iteration_buf.txt', 'wb') as f:
    pickle.dump(step_buf, f)
print(fid_buf)
min_fid = min(fid_buf)
min_fid_step = step_buf[fid_buf.index(min_fid)]
plt.close('all')
plt.plot(step_buf, fid_buf)
plt.scatter(min_fid_step, min_fid, color='r')
plt.annotate('({}, {})'.format(min_fid_step, min_fid), xy=(0.6, 0.95), xycoords='axes fraction')
plt.xlabel('Iterations')
plt.ylabel('FID')
plt.savefig('{}fid_plot.png'.format(generated_data_dir), dpi=None)

print(HEADER + 'Evaluating Reconstruction FID Score' + END_C)

fid_buf = []
step_buf = []

for i in range(80 * args.save_interval, args.training_steps, args.save_interval):
    tf.compat.v1.reset_default_graph()
    fid_val = evaluate_fid(image_dir=reconstructed_data_dir, prefix='reconstructed', dataset=args.dataset, step=i,
                           root_folder='./')
    print(i, fid_val)
    fid_buf.append(fid_val)
    step_buf.append(i)
with open(reconstructed_data_dir + 'fid_buf.txt', 'wb') as f:
    pickle.dump(fid_buf, f)
with open(reconstructed_data_dir + 'iteration_buf.txt', 'wb') as f:
    pickle.dump(step_buf, f)
print(fid_buf)
min_fid = min(fid_buf)
min_fid_step = step_buf[fid_buf.index(min_fid)]
plt.close('all')
plt.plot(step_buf, fid_buf)
plt.scatter(min_fid_step, min_fid, color='r')
plt.annotate('({}, {})'.format(min_fid_step, min_fid), xy=(0.6, 0.95), xycoords='axes fraction')
plt.xlabel('Iterations')
plt.ylabel('FID')
plt.savefig('{}fid_plot.png'.format(reconstructed_data_dir), dpi=None)
