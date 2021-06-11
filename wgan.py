from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import glob
import random
import collections

parser = argparse.ArgumentParser()
parser.add_argument("--mode",  choices=["train", "test", "export"])
parser.add_argument("--output_dir",  help="where to put output files")
parser.add_argument("--output_data",  help="where to put output files")
parser.add_argument("--input_dir",  help="The input file used to train or test")
parser.add_argument("--input_data",  help="The input file used to train or test")
parser.add_argument("--n", type=int, default=3, help="update summaries every summary_freq steps")
parser.add_argument("--seed", type=int)
parser.add_argument("--data_set", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="update summaries every summary_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")
parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate")
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 256

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")

def deprocess(image):
    with tf.name_scope("deprocess"):
        return (image+1)/2

def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))

def gen_conv(batch_input, out_channels):
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)

def gen_deconv(batch_input, out_channels):
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)
def lrelu(x, a):
    with tf.name_scope("lrelu"):
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)
def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))
def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, a.ngf)
        print(output.shape)
        layers.append(output)

    layer_specs = [
        a.ngf * 2, 
        a.ngf * 4,
        a.ngf * 8, 
        a.ngf * 8, 
        a.ngf * 8, 
        a.ngf * 8, 
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (a.ngf * 8, 0.5),   
        (a.ngf * 8, 0.5),   
        (a.ngf * 8, 0.5),   
        (a.ngf * 4, 0.5),   
        (a.ngf * 2, 0.0),   
        (a.ngf, 0.0),      
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)
            rectified = tf.nn.relu(input)
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def create_model(inputs, targets):
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = a.n
        layers = []

        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        with tf.variable_scope("layer_1"):
            convolved = discrim_conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)


        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = discrim_conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = discrim_conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        discrim_loss = tf.reduce_mean(predict_fake  - predict_real)

    with tf.name_scope("generator_loss"):
        gen_loss_GAN = - tf.reduce_mean(predict_fake)
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.RMSPropOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)
        clip_vars = [tf.assign(var, tf.clip_by_value(var, -0.02, 0.02)) for var in discrim_tvars]
        tuple_vars = tf.tuple(clip_vars, control_inputs=[discrim_train])

    with tf.name_scope("generator_train"):
        with tf.control_dependencies(tuple_vars):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.RMSPropOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )

def _parse_function(value):
    feature = tf.parse_single_example(value, features={
        "input": tf.FixedLenFeature((), tf.string),
        "target": tf.FixedLenFeature((), tf.string)
    })
    input = feature["input"]
    target = feature["target"]

    input_decoded = tf.decode_raw(input, tf.float32)
    target_decoded = tf.decode_raw(target, tf.float32)

    target_reshaped = tf.reshape(target_decoded, [128, 128, 1])
    input_reshaped = tf.reshape(input_decoded, [128, 128, 1])

    return input_reshaped, target_reshaped

def main():
    tf.reset_default_graph()  
    
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

    if a.input_dir is not None:
        input_path = glob.glob(os.path.join(a.input_dir, "*"))
        random.shuffle(input_path)
        dataset = tf.data.TFRecordDataset(input_path)
        
    if a.input_data is not None:
        dataset = tf.data.TFRecordDataset(a.input_data)

    
    dataset = dataset.map(_parse_function)
    if a.mode == "train":
        dataset = dataset.shuffle(buffer_size=10000).repeat(2).batch(a.batch_size)
    else:
        dataset = dataset.batch(a.batch_size)
        

    iterator = dataset.make_initializable_iterator()
    (target_batch, input_batch) = iterator.get_next()


    model = create_model(input_batch, target_batch) 
    
    inputs = deprocess(input_batch)
    targets = deprocess(target_batch)
    outputs = deprocess(model.outputs)

    
    converted_inputs = tf.image.convert_image_dtype(inputs, dtype=tf.uint8, saturate=True)
    converted_targets = tf.image.convert_image_dtype(targets, dtype=tf.uint8, saturate=True)
    converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)
    
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)


    with tf.name_scope("test_fetches"):
        test_fetches={
                "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="inputs"),
                "inputs_array": input_batch,
                "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="targets"),
                "target_array": target_batch,
                "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="outputs"),
                "outputs_array": model.outputs,
                "errors": tf.map_fn(tf.image.encode_png, tf.image.convert_image_dtype(tf.abs(outputs-targets), dtype=tf.uint8, saturate=True), dtype=tf.string, name="errors")
            }

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)


    saver = tf.train.Saver(max_to_keep=200)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    logdir = a.output_dir if ( a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session(config=config) as sess:
	
        sess.run(iterator.initializer)

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)
            print("loading model from checkpoint successfully")

       
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            input_real = []
            output_real = []
            target_real = []
            for step in range(max_steps):
                print(step)
                results = sess.run(test_fetches)
                input_real.append(results["inputs_array"])
                output_real.append(results["outputs_array"])
                target_real.append(results["target_array"])


            np.save(os.path.join(a.output_dir, "input.npy"), input_real)
            np.save(os.path.join(a.output_dir, "output.npy"), output_real)
            np.save(os.path.join(a.output_dir, "target.npy"), target_real)
        else:
            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options= None
                run_metadata = None
                
                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op
                    
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()                    

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(a.summary_freq):
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])
                    
                if should(a.trace_freq):
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" %results["global_step"])

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)
                if sv.should_stop():
                    break
main()
