import os
import time
import numpy as np
import tensorflow as tf


def assign_to_gpu(gpu=0, ps_dev="/device:CPU:0"):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op == "Variable":
            return ps_dev
        else:
            return "/gpu:%d" % gpu
    return _assign


def find_trainable_variables(key):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ".*{}.*".format(key))


def split(x, split_dim, split_sizes):
    n = len(list(x.get_shape()))
    dim_size = np.sum(split_sizes)
    assert int(x.get_shape()[split_dim]) == dim_size
    ids = np.cumsum([0] + split_sizes)
    ids[-1] = -1
    begin_ids = ids[:-1]

    ret = []
    for i in range(len(split_sizes)):
        cur_begin = np.zeros([n], dtype=np.int32)
        cur_begin[split_dim] = begin_ids[i]
        cur_end = np.zeros([n], dtype=np.int32) - 1
        cur_end[split_dim] = split_sizes[i]
        ret += [tf.slice(x, cur_begin, cur_end)]
    return ret


def load_from_checkpoint(saver, logdir):
    sess = tf.get_default_session()
    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt and ckpt.model_checkpoint_path:
        if os.path.isabs(ckpt.model_checkpoint_path):
            # Restores from checkpoint with absolute path.
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            # Restores from checkpoint with relative path.
            saver.restore(sess, os.path.join(logdir, ckpt.model_checkpoint_path))
        return True
    return False


class CheckpointLoader(object):
    def __init__(self, saver, global_step, logdir):
        self.saver = saver
        self.global_step_tensor = global_step
        self.logdir = logdir
        # TODO(rafal): make it restart-proof?
        self.last_global_step = 0

    def load_checkpoint(self):
        while True:
            if load_from_checkpoint(self.saver, self.logdir):
                global_step = int(self.global_step_tensor.eval())
                if global_step <= self.last_global_step:
                    print("Waiting for a new checkpoint...")
                    time.sleep(60)
                    continue
                print("Succesfully loaded model at step=%s." % global_step)
            else:
                print("No checkpoint file found. Waiting...")
                time.sleep(60)
                continue
            self.last_global_step = global_step
            return True


def average_grads(tower_grads):
    def average_dense(grad_and_vars):
        if len(grad_and_vars) == 1:
            return grad_and_vars[0][0]

        grad = grad_and_vars[0][0]
        for g, _ in grad_and_vars[1:]:
            grad += g
        return grad / len(grad_and_vars)

    def average_sparse(grad_and_vars):
        if len(grad_and_vars) == 1:
            return grad_and_vars[0][0]

        indices = []
        values = []
        for g, _ in grad_and_vars:
            indices += [g.indices]
            values += [g.values]
        indices = tf.concat(0, indices)
        values = tf.concat(0, values)
        return tf.IndexedSlices(values, indices, grad_and_vars[0][0].dense_shape)

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        if grad_and_vars[0][0] is None:
            grad = None
        elif isinstance(grad_and_vars[0][0], tf.IndexedSlices):
            grad = average_sparse(grad_and_vars)
        else:
            grad = average_dense(grad_and_vars)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def img_stretch(img):
    img = img.astype(np.float32)
    img -= np.min(img)
    img /= np.max(img) + 1e-12
    return img


def img_tile(imgs, aspect_ratio=1.0, tile_shape=None, border=1,
             border_color=0):
    """ Tile images in a grid.
    If tile_shape is provided only as many images as specified in tile_shape
    will be included in the output.
    """

    # Prepare images
    imgs = np.asarray(imgs)
    if imgs.ndim != 3 and imgs.ndim != 4:
        raise ValueError('imgs has wrong number of dimensions.')
    n_imgs = imgs.shape[0]

    # Grid shape
    img_shape = np.array(imgs.shape[1:3])
    if tile_shape is None:
        img_aspect_ratio = img_shape[1] / float(img_shape[0])
        aspect_ratio *= img_aspect_ratio
        tile_height = int(np.ceil(np.sqrt(n_imgs * aspect_ratio)))
        tile_width = int(np.ceil(np.sqrt(n_imgs / aspect_ratio)))
        grid_shape = np.array((tile_height, tile_width))
    else:
        assert len(tile_shape) == 2
        grid_shape = np.array(tile_shape)

    # Tile image shape
    tile_img_shape = np.array(imgs.shape[1:])
    tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

    # Assemble tile image
    tile_img = np.empty(tile_img_shape)
    tile_img[:] = border_color
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            img_idx = j + i * grid_shape[1]
            if img_idx >= n_imgs:
                # No more images - stop filling out the grid.
                break
            img = imgs[img_idx]
            yoff = (img_shape[0] + border) * i
            xoff = (img_shape[1] + border) * j
            tile_img[yoff:yoff + img_shape[0], xoff:xoff + img_shape[1], ...] = img

    return tile_img


# Fixes supervisor to start queue runners before initializing the model.
# TODO(rafal): Send a patch to main tensorflow repo.
class NotBuggySupervisor(tf.train.Supervisor):
    def prepare_or_wait_for_session(self, master="", config=None,
                                    wait_for_checkpoint=False,
                                    max_wait_secs=7200,
                                    start_standard_services=True):
        """Make sure the model is ready to be used.

        Create a session on 'master', recovering or initializing the model as
        needed, or wait for a session to be ready.  If running as the chief
        and `start_standard_service` is set to True, also call the session
        manager to start the standard services.

        Args:
          master: name of the TensorFlow master to use.  See the `tf.Session`
            constructor for how this is interpreted.
          config: Optional ConfigProto proto used to configure the session,
            which is passed as-is to create the session.
          wait_for_checkpoint: Whether we should wait for the availability of a
            checkpoint before creating Session. Defaults to False.
          max_wait_secs: Maximum time to wait for the session to become available.
          start_standard_services: Whether to start the standard services and the
            queue runners.

        Returns:
          A Session object that can be used to drive the model.
        """
        # For users who recreate the session with prepare_or_wait_for_session(), we
        # need to clear the coordinator's stop_event so that threads managed by the
        # coordinator can run.
        self._coord.clear_stop()

        if self._is_chief:
            sess, initialized = self._session_manager.recover_session(
                master, self.saver, checkpoint_dir=self._logdir,
                wait_for_checkpoint=wait_for_checkpoint,
                max_wait_secs=max_wait_secs, config=config)

            if start_standard_services:
                print("Starting queue runners")
                self.start_queue_runners(sess)

            if not initialized:
                if not self.init_op and not self._init_fn:
                    raise RuntimeError("Model is not initialized and no init_op or "
                                       "init_fn was given")
                if self.init_op:
                    sess.run(self.init_op, feed_dict=self._init_feed_dict)
                if self._init_fn:
                    self._init_fn(sess)
                not_ready = self._session_manager._model_not_ready(sess)
                if not_ready:
                    raise RuntimeError("Init operations did not make model ready.  "
                                       "Init op: %s, init fn: %s, error: %s"
                                       % (self.init_op.name, self._init_fn, not_ready))

            self._write_graph()
            if start_standard_services:
                self.start_standard_services(sess)
        else:
            sess = self._session_manager.wait_for_session(master,
                                                          config=config,
                                                          max_wait_secs=max_wait_secs)
            if start_standard_services:
                self.start_queue_runners(sess)
        return sess
