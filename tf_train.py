import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
from tf_utils.adamax import AdamaxOptimizer
from tf_utils.hparams import HParams
from tf_utils.common import img_stretch, img_tile
from tf_utils.common import assign_to_gpu, split, CheckpointLoader, average_grads, NotBuggySupervisor
from tf_utils.layers import conv2d, deconv2d, ar_multiconv2d, resize_nearest_neighbor
from tf_utils.distributions import DiagonalGaussian, discretized_logistic, compute_lowerbound, repeat
from tf_utils.data_utils import get_inputs, get_images
import tqdm

# settings
flags = tf.flags
flags.DEFINE_string("logdir", "/tmp/vae", "Logging directory.")
flags.DEFINE_string("hpconfig", "", "Overrides default hyper-parameters.")
flags.DEFINE_string("mode", "train", "Whether to run 'train' or 'eval' model.")
flags.DEFINE_integer("num_gpus", 8, "Number of GPUs used.")
FLAGS = flags.FLAGS


class IAFLayer(object):
    def __init__(self, hps, mode, downsample):
        self.hps = hps
        self.mode = mode
        self.downsample = downsample

    def up(self, input, **_):
        hps = self.hps
        h_size = hps.h_size
        z_size = hps.z_size
        stride = [2, 2] if self.downsample else [1, 1]

        with arg_scope([conv2d]):
            x = tf.nn.elu(input)
            x = conv2d("up_conv1", x, 2 * z_size + 2 * h_size, stride=stride)
            self.qz_mean, self.qz_logsd, self.up_context, h = split(x, 1, [z_size, z_size, h_size, h_size])

            h = tf.nn.elu(h)
            h = conv2d("up_conv3", h, h_size)
            if self.downsample:
                input = resize_nearest_neighbor(input, 0.5)
            return input + 0.1 * h

    def down(self, input):
        hps = self.hps
        h_size = hps.h_size
        z_size = hps.z_size

        with arg_scope([conv2d, ar_multiconv2d]):
            x = tf.nn.elu(input)
            x = conv2d("down_conv1", x, 4 * z_size + h_size * 2)
            pz_mean, pz_logsd, rz_mean, rz_logsd, down_context, h_det = split(x, 1, [z_size] * 4 + [h_size] * 2)

            prior = DiagonalGaussian(pz_mean, 2 * pz_logsd)
            posterior = DiagonalGaussian(rz_mean + self.qz_mean, 2 * (rz_logsd + self.qz_logsd))
            context = self.up_context + down_context

            if self.mode in ["init", "sample"]:
                z = prior.sample
            else:
                z = posterior.sample

            if self.mode == "sample":
                kl_cost = kl_obj = tf.zeros([hps.batch_size * hps.k])
            else:
                logqs = posterior.logps(z)
                x = ar_multiconv2d("ar_multiconv2d", z, context, [h_size, h_size], [z_size, z_size])
                arw_mean, arw_logsd = x[0] * 0.1, x[1] * 0.1
                z = (z - arw_mean) / tf.exp(arw_logsd)
                logqs += arw_logsd
                logps = prior.logps(z)

                kl_cost = logqs - logps

                if hps.kl_min > 0:
                    # [0, 1, 2, 3] -> [0, 1] -> [1] / (b * k)
                    kl_ave = tf.reduce_mean(tf.reduce_sum(kl_cost, [2, 3]), [0], keep_dims=True)
                    kl_ave = tf.maximum(kl_ave, hps.kl_min)
                    kl_ave = tf.tile(kl_ave, [hps.batch_size * hps.k, 1])
                    kl_obj = tf.reduce_sum(kl_ave, [1])
                else:
                    kl_obj = tf.reduce_sum(kl_cost, [1, 2, 3])
                kl_cost = tf.reduce_sum(kl_cost, [1, 2, 3])

            h = tf.concat(1, [z, h_det])
            h = tf.nn.elu(h)
            if self.downsample:
                input = resize_nearest_neighbor(input, 2)
                h = deconv2d("down_deconv2", h, h_size)
            else:
                h = conv2d("down_conv2", h, h_size)
            output = input + 0.1 * h
            return output, kl_obj, kl_cost


def get_default_hparams():
    return HParams(
        batch_size=16,        # Batch size on one GPU.
        eval_batch_size=100,  # Batch size for evaluation.
        num_gpus=8,           # Number of GPUs (effectively increases batch size).
        learning_rate=0.01,   # Learning rate.
        z_size=32,            # Size of z variables.
        h_size=160,           # Size of resnet block.
        kl_min=0.25,          # Number of "free bits/nats".
        depth=2,              # Number of downsampling blocks.
        num_blocks=2,         # Number of resnet blocks for each downsampling layer.
        k=1,                  # Number of samples for IS objective.
        dataset="cifar10",    # Dataset name.
        image_size=32,        # Image size.
    )


class CVAE1(object):
    def __init__(self, hps, mode, x=None):
        self.hps = hps
        self.mode = mode
        input_shape = [hps.batch_size * hps.num_gpus, 3, hps.image_size, hps.image_size]
        self.x = tf.placeholder(tf.uint8, shape=input_shape) if x is None else x
        self.m_trunc = []
        self.dec_log_stdv = tf.get_variable("dec_log_stdv", initializer=tf.constant(0.0))

        losses = []
        grads = []
        xs = tf.split(0, hps.num_gpus, self.x)
        opt = AdamaxOptimizer(hps.learning_rate)

        num_pixels = 3 * hps.image_size * hps.image_size
        for i in range(hps.num_gpus):
            with tf.device(assign_to_gpu(i)):
                m, obj, loss = self._forward(xs[i], i)
                losses += [loss]
                self.m_trunc += [m]

                # obj /= (np.log(2.) * num_pixels * hps.batch_size)
                if mode == "train":
                    grads += [opt.compute_gradients(obj)]

        self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.zeros_initializer,
                                           trainable=False)
        self.bits_per_dim = tf.add_n(losses) / (np.log(2.) * num_pixels * hps.batch_size * hps.num_gpus)

        if mode == "train":
            # add gradients together and get training updates
            grad = average_grads(grads)
            self.train_op = opt.apply_gradients(grad, global_step=self.global_step)
            tf.scalar_summary("model/bits_per_dim", self.bits_per_dim)
            tf.scalar_summary("model/dec_log_stdv", self.dec_log_stdv)
            self.summary_op = tf.merge_all_summaries()
        else:
            self.train_op = tf.no_op()

        if mode in ["train", "eval"]:
            with tf.name_scope(None):  # This is needed due to EMA implementation silliness.
                # keep track of moving average
                ema = tf.train.ExponentialMovingAverage(decay=0.999)
                self.train_op = tf.group(*[self.train_op, ema.apply(tf.trainable_variables())])
                self.avg_dict = ema.variables_to_restore()

    def _forward(self, x, gpu):
        hps = self.hps

        x = tf.to_float(x)
        x = tf.clip_by_value((x + 0.5) / 256.0, 0.0, 1.0) - 0.5

        # Input images are repeated k times on the input.
        # This is used for Importance Sampling loss (k is number of samples).
        data_size = hps.batch_size * hps.k
        x = repeat(x, hps.k)

        orig_x = x
        h_size = hps.h_size

        with arg_scope([conv2d, deconv2d], init=(self.mode == "init")):
            layers = []
            for i in range(hps.depth):
                layers.append([])
                for j in range(hps.num_blocks):
                    downsample = (i > 0) and (j == 0)
                    layers[-1].append(IAFLayer(hps, self.mode, downsample))

            h = conv2d("x_enc", x, h_size, [5, 5], [2, 2])  # -> [16, 16]
            for i, layer in enumerate(layers):
                for j, sub_layer in enumerate(layer):
                    with tf.variable_scope("IAF_%d_%d" % (i, j)):
                        h = sub_layer.up(h)

            # top->down
            self.h_top = h_top = tf.get_variable("h_top", [h_size], initializer=tf.zeros_initializer)
            h_top = tf.reshape(h_top, [1, -1, 1, 1])
            h = tf.tile(h_top, [data_size, 1, hps.image_size / 2 ** len(layers), hps.image_size / 2 ** len(layers)])
            kl_cost = kl_obj = 0.0

            for i, layer in reversed(list(enumerate(layers))):
                for j, sub_layer in reversed(list(enumerate(layer))):
                    with tf.variable_scope("IAF_%d_%d" % (i, j)):
                        h, cur_obj, cur_cost = sub_layer.down(h)
                        kl_obj += cur_obj
                        kl_cost += cur_cost

                        if self.mode == "train" and gpu == hps.num_gpus - 1:
                            tf.scalar_summary("model/kl_obj_%02d_%02d" % (i, j), tf.reduce_mean(cur_obj))
                            tf.scalar_summary("model/kl_cost_%02d_%02d" % (i, j), tf.reduce_mean(cur_cost))

            x = tf.nn.elu(h)
            x = deconv2d("x_dec", x, 3, [5, 5])
            x = tf.clip_by_value(x, -0.5 + 1 / 512., 0.5 - 1 / 512.)

        log_pxz = discretized_logistic(x, self.dec_log_stdv, sample=orig_x)
        obj = tf.reduce_sum(kl_obj - log_pxz)

        if self.mode == "train" and gpu == hps.num_gpus - 1:
            tf.scalar_summary("model/log_pxz", -tf.reduce_mean(log_pxz))
            tf.scalar_summary("model/kl_obj", tf.reduce_mean(kl_obj))
            tf.scalar_summary("model/kl_cost", tf.reduce_mean(kl_cost))

        loss = tf.reduce_sum(compute_lowerbound(log_pxz, kl_cost, hps.k))
        return x, obj, loss


def run(hps):
    with tf.variable_scope("model") as vs:
        x = get_inputs(hps.dataset, "train", hps.batch_size * FLAGS.num_gpus, hps.image_size)

        hps.num_gpus = 1
        init_x = x[:hps.batch_size, :, :, :]
        init_model = CVAE1(hps, "init", init_x)

        vs.reuse_variables()
        hps.num_gpus = FLAGS.num_gpus
        model = CVAE1(hps, "train", x)

    saver = tf.train.Saver()

    total_size = 0
    for v in tf.trainable_variables():
        total_size += np.prod([int(s) for s in v.get_shape()])
    print("Num trainable variables: %d" % total_size)

    init_op = tf.initialize_all_variables()

    def init_fn(ses):
        print("Initializing parameters.")
        # XXX(rafal): TensorFlow bug?? Default initializer should handle things well..
        ses.run(init_model.h_top.initializer)
        ses.run(init_op)
        print("Initialized!")

    sv = NotBuggySupervisor(is_chief=True,
                            logdir=FLAGS.logdir + "/train",
                            summary_op=None,  # Automatic summaries don"t work with placeholders.
                            saver=saver,
                            global_step=model.global_step,
                            save_summaries_secs=30,
                            save_model_secs=0,
                            init_op=None,
                            init_fn=init_fn)

    print("starting training")
    local_step = 0
    begin = time.time()

    config = tf.ConfigProto(allow_soft_placement=True)
    with sv.managed_session(config=config) as sess:
        print("Running first iteration!")
        while not sv.should_stop():
            fetches = [model.bits_per_dim, model.global_step, model.dec_log_stdv, model.train_op]

            should_compute_summary = (local_step % 20 == 19)
            if should_compute_summary:
                fetches += [model.summary_op]

            fetched = sess.run(fetches)

            if should_compute_summary:
                sv.summary_computed(sess, fetched[-1])

            if local_step < 10 or should_compute_summary:
                print("Iteration %d, time = %.2fs, train bits_per_dim = %.4f, dec_log_stdv = %.4f" % (
                      fetched[1], time.time() - begin, fetched[0], fetched[2]))
                begin = time.time()
            if np.isnan(fetched[0]):
                print("NAN detected!")
                break
            if local_step % 100 == 0:
                saver.save(sess, sv.save_path, global_step=sv.global_step, write_meta_graph=False)

            local_step += 1
        sv.stop()


def run_eval(hps, mode):
    hps.num_gpus = 1
    hps.batch_size = hps.eval_batch_size

    with tf.variable_scope("model") as vs:
        model = CVAE1(hps, "eval")
        vs.reuse_variables()
        sample_model = CVAE1(hps, "sample")

    saver = tf.train.Saver(model.avg_dict)
    # Use only 4 threads for the evaluation.
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=4,
                            inter_op_parallelism_threads=4)
    sess = tf.Session(config=config)
    sw = tf.train.SummaryWriter(FLAGS.logdir + "/" + FLAGS.mode, sess.graph)
    ckpt_loader = CheckpointLoader(saver, model.global_step, FLAGS.logdir + "/train")

    with sess.as_default():
        dataset = get_images(hps.dataset, mode[5:], hps.image_size)
        assert dataset.n % hps.batch_size == 0
        epoch_size = int(dataset.n / hps.batch_size)

        while ckpt_loader.load_checkpoint():
            global_step = ckpt_loader.last_global_step

            dataset.shuffle()
            summary = tf.Summary()
            all_bits_per_dim = []
            for _ in tqdm.trange(epoch_size):
                all_bits_per_dim += [sess.run(model.bits_per_dim, {model.x: dataset.next_batch(hps.batch_size)})]

            average_bits = float(np.mean(all_bits_per_dim))
            print("Step: %d Score: %.3f" % (global_step, average_bits))
            summary.value.add(tag='eval_bits_per_dim', simple_value=average_bits)

            if hps.k == 1:
                # show reconstructions from the model
                total_samples = 36
                num_examples = 0
                imgs_inputs = np.zeros([total_samples / 2, hps.image_size, hps.image_size, 3], np.float32)
                imgs_recs = np.zeros([total_samples / 2, hps.image_size, hps.image_size, 3], np.float32)
                while num_examples < total_samples / 2:
                    batch = dataset.next_batch(hps.batch_size)
                    sample_x = sess.run(model.m_trunc[0], {model.x: batch})
                    batch_bhwc = np.transpose(batch, (0, 2, 3, 1))
                    img_bhwc = np.transpose(sample_x, (0, 2, 3, 1))

                    if num_examples + hps.batch_size > total_samples / 2:
                        cur_examples = total_samples / 2 - num_examples
                    else:
                        cur_examples = hps.batch_size

                    imgs_inputs[num_examples:num_examples + cur_examples, ...] = img_stretch(batch_bhwc[:cur_examples, ...])
                    imgs_recs[num_examples:num_examples + cur_examples, ...] = img_stretch(img_bhwc[:cur_examples, ...])
                    num_examples += cur_examples

                imgs_to_plot = np.zeros([total_samples, hps.image_size, hps.image_size, 3], np.float32)
                imgs_to_plot[::2, ...] = imgs_inputs
                imgs_to_plot[1::2, ...] = imgs_recs
                imgs = img_tile(imgs_to_plot, aspect_ratio=1.0, border=0).astype(np.float32)
                imgs = np.expand_dims(imgs, 0)
                im_summary = tf.image_summary("reconstructions", imgs, 1)
                summary.MergeFromString(sess.run(im_summary))

                # generate samples from the model
                num_examples = 0
                imgs_to_plot = np.zeros([total_samples, hps.image_size, hps.image_size, 3], np.float32)
                while num_examples < total_samples:
                    sample_x = sess.run(sample_model.m_trunc[0])
                    img_bhwc = img_stretch(np.transpose(sample_x, (0, 2, 3, 1)))

                    if num_examples + hps.batch_size > total_samples:
                        cur_examples = total_samples - num_examples
                    else:
                        cur_examples = hps.batch_size

                    imgs_to_plot[num_examples:num_examples+cur_examples, ...] = img_stretch(img_bhwc[:cur_examples, ...])
                    num_examples += cur_examples

                imgs = img_tile(imgs_to_plot, aspect_ratio=1.0, border=0).astype(np.float32)
                imgs = np.expand_dims(imgs, 0)
                im_summary = tf.image_summary("samples", imgs, 1)
                summary.MergeFromString(sess.run(im_summary))

            sw.add_summary(summary, global_step)
            sw.flush()


def main(_):
    hps = get_default_hparams().parse(FLAGS.hpconfig)

    if FLAGS.mode == "train":
        run(hps)
    else:
        run_eval(hps, FLAGS.mode)


if __name__ == "__main__":
    tf.app.run()
