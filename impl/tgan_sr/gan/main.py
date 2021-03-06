# pylint: disable=line-too-long, invalid-name

"""Defines the Transformer GAN architecture and its custom training loop"""

from functools import partial
import os

import tensorflow as tf

from tgan_sr.transformer import lr_schedules
from tgan_sr.gan.utils import CreatedBuffer, parse_score, parse_score_math, get_corrects
from tgan_sr.gan.base import TransformerGenerator, TransformerCritic
from tgan_sr.utils.utils import increment



class TransformerGAN(tf.keras.Model):
    """Transformer GAN architecture. Defines a self-contained model with optimizers, losses and custom training loop"""

    def __init__(self, params, vocab, log_dir):
        """
        params : hyperparameters for the architecture, specified in train/gan.py
        vocab : input vocabulary. Required for decoding examples
        log_dir : tensorboard summary logger output directory
        """
        super().__init__()
        self.__dict__['params'] = params # do not save in checkpoint
        self.generate_confusion = params['gan_generate_confusion']
        self.inherent_class_loss = self.generate_confusion
        self.__dict__['objectives'] = params['objectives'].split(',') # do not save in checkpoint
        if params['gan_incremental_learning_mode']:
            self.created_buffer = CreatedBuffer(params)
        self.warnings = {}
        self.vocab = vocab
        self.tb_writer = tf.summary.create_file_writer(log_dir + '/train_custom')
        self.epoch_steps = 0
        self.test_steps = 0
        self.epoch = 0
        self.total_steps = 0
        self.last_ana = None

        # critic and generator
        proc_logits_fn = partial(proc_logits, normalize=True, tau=1, sample=False, calc_entropy=False) # processing function for generator output
        self.generator = TransformerGenerator(params, proc_logits_fn=proc_logits_fn)
        self.critic = TransformerCritic(params, sigmoid=False)

        # learning rate and losses
        self.learning_rate = lr_schedules.TransformerSchedule(params['d_embed_enc'], warmup_steps=params['warmup_steps']) # dynamic learning rate, only used in pure classifier setting (no GAN)
        learning_rate = 1e-4 # constant learning rate, used for (W)GANs
        if self.params['gan_class_loss'] == 'crossentropy':
            self.class_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0)
        elif self.params['gan_class_loss'] == 'hinge':
            self.class_loss = tf.keras.losses.Hinge()
        self.crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # optimizers
        if 'class' in self.objectives and not 'gan' in self.objectives and not params['gan_force_constant_lr']:
            # pure classifier setting, use specilized Transformer optimizer settings
            self.opti_c = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        else:
            # GAN setting, use simpler optimizer
            self.opti_c = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0., beta_2=0.9)
        if 'gan' in self.objectives:
            self.opti_g = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0., beta_2=0.9)


    def input_noise(self, dimensions_per_position, batch_size_or_mask, len_mode='uniform', min_len=1, max_len=None, random_dist='uniform'):
        """generate a tensor of random noise (z) to be input to the generator. dimension bs ?? l ?? dimensions_per_position

        dimensions_per_position : currently only using value 1
        batch_size_or_mask : if len_mode=copy, provide a positive boolean mask to be copied here. else the batch_size
        len_mode : copy (from an existing mask) or uniform (sample lengths uniformly between min_len and max_len)
        max_len : optional, if None will use maximal allowed length from parameters
        random_dist : distribution for z itself. currently, only uniform implemented
        """
        if max_len is None:
            max_len = self.params['max_encode_length']
        if len_mode == 'copy':
            batch_size, max_len = tf.shape(batch_size_or_mask)
            positive_mask = tf.identity(batch_size_or_mask)
            lengths = tf.reduce_sum(tf.cast(positive_mask, tf.int32), axis=1)
        else:
            batch_size = batch_size_or_mask
        if random_dist == 'uniform':
            z = tf.random.uniform((batch_size, max_len, dimensions_per_position), 0, 1, dtype=tf.float32)
        if len_mode != 'copy':
            if len_mode == 'uniform':
                lengths = tf.random.uniform((batch_size,), min_len, max_len, dtype=tf.int32) + 1
            actual_max_len = max_len
            z = z[:, :actual_max_len, :]
            range_ = tf.reshape(tf.range(0, actual_max_len), (1, actual_max_len))
            positive_mask = range_ < lengths[:, tf.newaxis]
            assert all(tf.shape(positive_mask) == (batch_size, actual_max_len))
        return z, positive_mask


    def encode_real(self, x, step=None):
        """transform integer vocabulary indices into one_hot representation

        x : bs ?? l integer tensor
        step : unused
        returns bs ?? l ?? |V| one-hot tensor and bs ?? l boolean positive mask
        """
        sigma = self.params['gan_sigma_real']
        real_positive_mask = x != self.params['input_pad_id']
        real_samples_cont = tf.one_hot(x, self.params['input_vocab_size'], axis=-1)
        if sigma == 0:
            return real_samples_cont, real_positive_mask
        step_sigma = sigma
        real_samples_cont += tf.math.abs(tf.random.normal(tf.shape(real_samples_cont), mean=0, stddev=step_sigma))
        real_samples_cont /= tf.reduce_sum(real_samples_cont, axis=-1, keepdims=True)
        return real_samples_cont, real_positive_mask


    def hard_decode(self, x, pos_mask, full=True, as_list=True):
        """apply mask to integer predictions and decode to string tokens

        x : n ?? l integer tensor
        pos_mask : n ?? l boolean positive mask
        full, as_list : passed to vocab.decode()
        """
        return [self.vocab.decode(list(q[:tf.reduce_sum(tf.cast(m, tf.int32)).numpy()].numpy()), full=full, as_list=as_list) for q, m in zip(x, pos_mask)]


    def train_step(self, data):
        x, y_target = data
        x, _ = x

        batch_size, seq_len = tf.shape(x)
        iterations = self.total_steps
        critic_variables, generator_variables = None, None
        metrics = {}

        # Incremental learning, created buffer
        if self.params['gan_incremental_learning_mode']:
            zero_step, full_step = self.params['gan_incremental_usage_zero_step'], self.params['gan_incremental_usage_full_step'] 
            percentage_from_buffer = (self.total_steps - zero_step) / (full_step - zero_step)
            percentage_from_buffer = min(max(0, percentage_from_buffer), 1)
            items_from_buffer = tf.cast(self.params['batch_size'] * percentage_from_buffer, tf.int32)
            if self.created_buffer.buffer_items < items_from_buffer:
                if not 'created_buffer_health' in self.warnings:
                    self.warnings['created_buffer_health'] = f'Created buffer too low! Requested {items_from_buffer} from {self.created_buffer.buffer_items}'
                items_from_buffer = self.created_buffer.buffer_items
            if items_from_buffer > 0:
                from_buffer = self.created_buffer.get(items_from_buffer)
                x_from_buffer = from_buffer[:, :-1]
                y_target_from_buffer = from_buffer[:, -1:]
                x = tf.concat([x_from_buffer, x[items_from_buffer:]], axis=0)
                y_target = tf.concat([y_target_from_buffer, y_target[items_from_buffer:]], axis=0)
            with self.tb_writer.as_default(step=iterations):
                tf.summary.scalar('4extra/4created_buffer_percentage', percentage_from_buffer)
                tf.summary.scalar('4extra/4created_buffer_total_items', self.created_buffer.total_items)


        x_soft, real_positive_mask = self.encode_real(x, step=self.total_steps)
        y_t = tf.squeeze(tf.cast(y_target == 2, tf.float32))    
        for c_step in range(self.params['gan_critic_steps']):
            with tf.GradientTape() as real_tape:
                ??_real = x_soft
                pred_raw = self.critic(??_real, real_positive_mask, training=True)

                # loss computation
                if 'gan' in self.objectives:
                    pred_logits_gan_real = pred_raw[:, 1]
                    pred_score_real = tf.nn.sigmoid(tf.clip_by_value(pred_logits_gan_real, -10., 10.))
                    if self.params['gan_critic_target_fn'] == 'logits':
                        critic_loss_real = pred_logits_gan_real
                        assert self.params['gan_critic_target_mode'] == 'direct'
                    elif self.params['gan_critic_target_fn'].startswith('sigmoid'):
                        critic_loss_real = pred_score_real
                    else:
                        critic_loss_real = 0 # ..
                    if self.params['gan_critic_target_fn'].endswith('log'):
                        critic_loss_real = tf.math.log(critic_loss_real)
                    critic_loss_real = - tf.reduce_mean(critic_loss_real)
                    crossentropy_loss = self.crossentropy(tf.ones_like(pred_logits_gan_real), tf.clip_by_value(pred_logits_gan_real, -10., 10.))
                    if self.params['gan_critic_target_fn'] == 'crossentropy':
                        critic_loss_real = crossentropy_loss
                else:
                    critic_loss_real = 0
                pred_logits_class_real = pred_raw[:, 0]
                if 'class' in self.objectives or self.inherent_class_loss:
                    class_loss_real = self.class_loss(y_t, tf.clip_by_value(pred_logits_class_real, -10., 10.))
                    critic_loss_real += class_loss_real * self.params['gan_objweight_class']

            critic_variables = self.critic.trainable_variables if critic_variables is None  else critic_variables
            _vars = critic_variables
            _grads = real_tape.gradient(critic_loss_real, _vars)
            critic_grads, _grads = _grads[:len(critic_variables)], _grads[len(critic_variables):]
            assert_finite(critic_grads, 'Critic real grads')

            # update metrics
            if 'class' in self.objectives or self.inherent_class_loss:
                increment(metrics, 'class_loss', class_loss_real)
                increment(metrics, 'class_acc', tf.keras.metrics.binary_accuracy(y_t, tf.nn.sigmoid(pred_logits_class_real)))
                real_class_prob = tf.nn.sigmoid(pred_logits_class_real)
                increment(metrics, 'class_entropy', -tf.reduce_mean(real_class_prob * tf.math.log(real_class_prob) + (1-real_class_prob) * tf.math.log(1-real_class_prob)))
                if self.params['gan_incremental_learning_mode']:
                    increment(metrics, 'class_acc_from_buffer', tf.keras.metrics.binary_accuracy(y_t[:items_from_buffer], tf.nn.sigmoid(pred_logits_class_real[:items_from_buffer])))
                    increment(metrics, 'class_acc_from_dataset', tf.keras.metrics.binary_accuracy(y_t[items_from_buffer:], tf.nn.sigmoid(pred_logits_class_real[items_from_buffer:])))

            if 'gan' in self.objectives:
                mean_pred_score_real = tf.reduce_mean(pred_score_real)
                if self.params['gan_critic_target_fn'] == 'logits':
                    mean_logits_real = tf.reduce_mean(pred_logits_gan_real)
                    increment(metrics, 'logits_real', mean_logits_real)
                else:
                    increment(metrics, 'score_real', mean_pred_score_real)
                    increment(metrics, 'crossentropy_real', crossentropy_loss)


            if 'gan' in self.objectives:
                # Generated samples
                if self.params['gan_copy_shape_critic']:
                    z, generated_positive_mask = self.input_noise(1, real_positive_mask, len_mode='copy')
                else:
                    z, generated_positive_mask = self.input_noise(1, batch_size, len_mode='uniform')

                ??_gen = self.generator(z, generated_positive_mask, training=True)
                ??_gen_train = ??_gen
                gen_mask_train = generated_positive_mask

                # Generated samples, critic training
                with tf.GradientTape() as fooled_tape:
                    pred_raw = self.critic(??_gen_train, gen_mask_train, training=True)
                    pred_logits_gen = pred_raw[:, 1]
                    pred_score_gen = tf.nn.sigmoid(tf.clip_by_value(pred_logits_gen, -10., 10.))

                    # loss computation
                    if self.params['gan_critic_target_fn'] == 'logits':
                        critic_loss_gen = pred_logits_gen
                        assert self.params['gan_critic_target_mode'] == 'direct'
                    elif self.params['gan_critic_target_fn'].startswith('sigmoid'):
                        critic_loss_gen = pred_score_gen
                        if self.params['gan_critic_target_mode'] == 'one-minus':
                            critic_loss_gen = 1 - critic_loss_gen
                    else:
                        critic_loss_gen = 0 # ..
                    if self.params['gan_critic_target_fn'].endswith('log'):
                        critic_loss_gen = tf.math.log(critic_loss_gen)
                    if self.params['gan_critic_target_mode'] == 'one-minus':
                        critic_loss_gen = - critic_loss_gen # - again because of 1 - before
                    critic_loss_gen = tf.reduce_mean(critic_loss_gen)
                    crossentropy_loss = self.crossentropy(tf.zeros_like(pred_logits_gen), tf.clip_by_value(pred_logits_gen, -10., 10.))
                    if self.params['gan_critic_target_fn'] == 'crossentropy':
                        critic_loss_gen = crossentropy_loss
                    pred_logits_class_gen = pred_raw[:, 0]

                critic_grads_ = fooled_tape.gradient(critic_loss_gen, critic_variables, unconnected_gradients='none')
                assert_finite(critic_grads_, 'Critic generated grads')
                critic_grads = [a + b for a, b in zip(critic_grads, critic_grads_)]

                # update metrics
                mean_logits_gen = tf.reduce_mean(pred_logits_gen)
                if self.params['gan_critic_target_fn'] == 'logits':
                    increment(metrics, 'logits_gen', tf.reduce_mean(mean_logits_gen))
                    increment(metrics, 'wasserstein', mean_logits_real - mean_logits_gen)
                mean_pred_score_gen = tf.reduce_mean(pred_score_gen)
                if self.params['gan_critic_target_fn'] != 'logits':
                    increment(metrics, 'score_gen', mean_pred_score_gen)
                    increment(metrics, 'crossentropy_gen', crossentropy_loss)
                    increment(metrics, 'crossentropy_genalt', self.crossentropy(tf.ones_like(pred_logits_gen), pred_logits_gen))

                if self.inherent_class_loss:
                    gen_class_prob = tf.nn.sigmoid(tf.clip_by_value(pred_logits_class_gen, -10., 10.))
                    if self.generate_confusion:
                        increment(metrics, 'genclass_entropy', -tf.reduce_mean(gen_class_prob * tf.math.log(gen_class_prob) + (1-gen_class_prob) * tf.math.log(1-gen_class_prob)))


                # Gradient penalty
                if self.params['gan_intgrad_method'] == 'none':
                    assert self.params['gan_gradient_penalty'] == 0
                else:
                    # Interleaved samples
                    gen_seq_len = tf.shape(??_gen)[1]
                    len_diff = (seq_len - gen_seq_len).numpy()
                    assert self.params['gan_copy_shape_critic']
                    if self.params['gan_copy_shape_critic']:
                        assert len_diff == 0
                        assert tf.math.reduce_all(real_positive_mask == generated_positive_mask)
                    # Uniform line samples
                    eps_lines = tf.random.uniform((batch_size,), 0, 1, dtype=tf.float32)[:, tf.newaxis, tf.newaxis]
                    ??_interleaved_uniform = eps_lines * ??_real + (1 - eps_lines) * ??_gen
                    interleaved_mask = tf.logical_and(real_positive_mask, generated_positive_mask)
                    if self.params['gan_intgrad_method'] == 'uniform':
                        ??_interleaved = ??_interleaved_uniform

                    def calc_input_gradients(??_interleaved):
                        with tf.GradientTape(watch_accessed_variables=False) as inner_penalty_tape:
                            inner_penalty_tape.watch(??_interleaved)
                            pred_raw = self.critic(??_interleaved, interleaved_mask, training=True)
                            pred_interleaved = pred_raw[:, 1]
                            # note: no mean here, since each output only depends on one input. Gradient is for each input separately.
                        grad_interleaved = inner_penalty_tape.gradient(pred_interleaved, ??_interleaved)
                        len_grad = tf.reduce_sum(grad_interleaved**2, axis=-1)
                        len_grad *= tf.cast(interleaved_mask, tf.float32)
                        len_grad = tf.math.sqrt(tf.reduce_sum(len_grad, axis=-1)) # one batch of scalars
                        return len_grad

                    if self.params['gan_gradient_penalty'] > 0:
                        with tf.GradientTape() as penalty_tape:
                            len_grad = calc_input_gradients(??_interleaved)
                            loss_gradient_penalty = self.params['gan_gradient_penalty'] * (len_grad - self.params['gan_intgrad_target']) ** 2
                            loss_gradient_penalty = tf.reduce_mean(loss_gradient_penalty)
                        critic_grads_ = penalty_tape.gradient(loss_gradient_penalty, critic_variables, unconnected_gradients='zero') # careful!
                        assert_finite(critic_grads_, 'Critic GP grads')
                        critic_grads = [a + b for a, b in zip(critic_grads, critic_grads_)]
                    else:
                        len_grad = calc_input_gradients(??_interleaved)
                    if self.params['gan_intgrad_method'] == 'uniform':
                        increment(metrics, 'intgrad_len_uniform', tf.reduce_mean(len_grad))
                # -- end GAN critic objective

            ## Apply gradients!
            if critic_variables is not None:
                self.opti_c.apply_gradients(zip(critic_grads, critic_variables))
            # -- end critic training loop
        for k in metrics:
            metrics[k] /= self.params['gan_critic_steps']


        # Generator training #
        if self.params['gan_copy_shape_generator']:
            z, generated_positive_mask = self.input_noise(1, real_positive_mask, len_mode='copy')
        else:
            z, generated_positive_mask = self.input_noise(1, batch_size)

        if 'gan' in self.objectives:
            with tf.GradientTape() as generator_tape:
                ??_gen = self.generator(z, generated_positive_mask, training=True)

                pred_raw = self.critic(??_gen, generated_positive_mask, training=True)
                pred_logits_gen = pred_raw[:, 1]
                pred_score_gen = tf.nn.sigmoid(pred_logits_gen)
                mean_pred_score_gen = tf.reduce_mean(pred_score_gen)

                # loss computation
                if self.params['gan_generator_target_fn'] == 'logits':
                    generator_loss = pred_logits_gen
                    assert self.params['gan_generator_target_mode'] == 'direct'
                elif self.params['gan_generator_target_fn'].startswith('sigmoid'):
                    generator_loss = pred_score_gen
                    if self.params['gan_generator_target_mode'] == 'one-minus':
                        generator_loss = 1 - generator_loss
                else:
                    generator_loss = 0
                if self.params['gan_generator_target_fn'].endswith('log'):
                    generator_loss = tf.math.log(generator_loss)
                if self.params['gan_generator_target_mode'] == 'direct':
                    generator_loss = - generator_loss
                generator_loss = tf.reduce_mean(generator_loss)
                crossentropy_loss = self.crossentropy(tf.ones_like(pred_logits_gen), pred_logits_gen)
                if self.params['gan_generator_target_fn'] == 'crossentropy':
                    generator_loss = crossentropy_loss
                pred_logits_class_gen = pred_raw[:, 0]
                if self.generate_confusion and self.total_steps >= self.params['gan_delay_confusion_steps']:
                    gen_class_prob = tf.nn.sigmoid(pred_logits_class_gen) #tf.clip_by_value(pred_logits_class_gen, -10., 10.))
                    gen_class_neg_entropy = gen_class_prob * tf.math.log(tf.clip_by_value(gen_class_prob, 1e-20, 1))
                    gen_class_neg_entropy += (1-gen_class_prob) * tf.math.log(tf.clip_by_value(1-gen_class_prob, 1e-20, 1))
                    if self.params['gan_confusion_loss'] == 'entropy':
                        gen_class_confusion_loss = gen_class_neg_entropy
                    elif self.params['gan_confusion_loss'] == 'mse':
                        gen_class_confusion_loss = pred_logits_class_gen ** 2
                    elif self.params['gan_confusion_loss'] == 'mae':
                        gen_class_confusion_loss = tf.math.abs(pred_logits_class_gen)
                    generator_loss += self.params['gan_objweight_confusion'] * tf.reduce_mean(gen_class_confusion_loss)

            generator_variables = self.generator.trainable_variables
            generator_grads = generator_tape.gradient(generator_loss, generator_variables)
            assert_finite(generator_grads, 'Generator grads')

            # Apply!
            self.opti_g.apply_gradients(zip(generator_grads,  generator_variables))
            if self.params['gan_generator_target_fn'] != 'logits':
                metrics['score_gen_alt'] = mean_pred_score_gen
            metrics['min_logits_gen'] = tf.reduce_min(pred_logits_gen)


            # Analyze generated samples
            if (self.total_steps % self.params['gan_trainsteps_infer_interval'] == 0):
                num_analyze = min(tf.shape(??_gen)[0].numpy(), 150)
                generated_tokens, ana, _ = self.get_predictions(num_analyze, ??_gen, generated_positive_mask)
                ana.update(self.analyze_generated(generated_tokens))

                # full batch here because feeding to critic
                generated_hards = proc_logits(??_gen, generated_positive_mask, sample=True, tau=0, calc_entropy=False)
                hard_gen_soft, _ = self.encode_real(generated_hards)
                pred_hard_gen = self.critic(hard_gen_soft, generated_positive_mask, training=True) # training=True for comparable scores
                pred_gan_logits_hard_gen = pred_hard_gen[:, 1]
                if self.params['gan_generator_target_fn'] == 'logits':
                    ana['logits_gen_hard'] = tf.reduce_mean(pred_gan_logits_hard_gen)
                else:
                    ana['score_gen_hard'] = tf.reduce_mean(tf.nn.sigmoid(pred_gan_logits_hard_gen))
                if self.generate_confusion:
                    hard_class_logits = pred_hard_gen[:, 0]
                    hard_class_prob = tf.nn.sigmoid(tf.clip_by_value(hard_class_logits, -10., 10.))
                    increment(metrics, 'genclass_logits_hard_mean', tf.reduce_mean(hard_class_logits))
                    entropies = -(hard_class_prob * tf.math.log(hard_class_prob) + (1-hard_class_prob) * tf.math.log(1-hard_class_prob))
                    ana['genclass_entropy_hard'] = tf.reduce_mean(entropies)

                ana = {k:v for k,v in ana.items() if v is not None}
                self.last_ana = ana
            else:
                num_analyze = 0
            metrics.update(self.last_ana)

            if (self.epoch_steps % 50 == 49):
                num_print = 3
                generated_tokens, ana, generated_hards = self.get_predictions(num_print, ??_gen, generated_positive_mask)
                join_str = ' ' if self.params['problem'] == 'math' else ''
                print(f'step {self.epoch_steps+1:3d}: ' + ', '.join([join_str.join(q) for q in generated_tokens]))

            # -- end GAN generator objective

        if not 'gan' in self.objectives and self.params['gan_process_generated_samples']:
            ??_gen = self.generator(z, generated_positive_mask, training=True)

        # Processing of generated samples: check, solve, save
        if (self.total_steps % self.params['gan_trainsteps_process_interval'] == 0) and self.params['gan_process_generated_samples']:
            generated_hards = proc_logits(??_gen, generated_positive_mask, sample=True, tau=0, calc_entropy=False)
            hard_gen_soft, _ = self.encode_real(generated_hards)
            pred_hard_gen = self.critic(hard_gen_soft, generated_positive_mask, training=True) # training=True damit ??hnliche scores
            hard_class_prob = tf.nn.sigmoid(pred_hard_gen[:, 0])
            entropies = -(hard_class_prob * tf.math.log(hard_class_prob) + (1-hard_class_prob) * tf.math.log(1-hard_class_prob))
            candidate_mask = entropies > self.params['gan_filter_generated_entropy_threshold']
            candidate_hards, candidate_positive_mask = tf.boolean_mask(generated_hards, candidate_mask), tf.boolean_mask(generated_positive_mask, candidate_mask)
            entropies = tf.boolean_mask(entropies, candidate_mask)
            candidate_tokens = self.hard_decode(candidate_hards, candidate_positive_mask)
            if self.params['gan_save_candidate_samples']:
                candidate_strings = [''.join(q) for q in candidate_tokens]
                with open(os.path.join(self.params['job_dir'], self.params['run_name'], 'generated_samples.txt'), 'a') as save_file:
                    save_file.write('\n'.join(candidate_strings))
            solved_indices_sat, _, solved_indices_unsat, _ = get_corrects(candidate_tokens, self.params, self.total_steps, entropies=entropies.numpy())
            if self.params['gan_incremental_learning_mode']:
                solved_indices_sat = tf.stack(solved_indices_sat)
                solved_indices_unsat = tf.stack(solved_indices_unsat)
                labels = tf.concat([tf.ones((len(solved_indices_sat),), dtype=tf.int32) * 2, tf.ones((len(solved_indices_unsat),), dtype=tf.int32) * 1], axis=0) # hardcoded :/
                solved_indices = tf.concat([solved_indices_sat, solved_indices_unsat], axis=0)
                if solved_indices.shape[0] > 0: # if we have any indices at all
                    candidate_hards = tf.where(candidate_positive_mask, candidate_hards, tf.ones_like(candidate_hards, dtype=tf.int32)* self.params['input_pad_id'])
                    to_save = tf.concat([tf.gather(candidate_hards, solved_indices, axis=0), labels[:, tf.newaxis]], axis=1) # save label and input together :)
                    self.created_buffer.update(to_save)


        with self.tb_writer.as_default(step=iterations): #pylint:disable=not-context-manager
            name_mapping = {'score_real' : '1critic/1score_real', 'score_gen' : '1critic/1score_gen', 'class_acc' : '1critic/2class_acc',
                'intgrad_len_uniform' : '1critic/3intgrad_len_uniform',
                'seq_entropy' : '1generator/1seq_entropy', 'parse_fragments' : '1generator/2parse_fragments', 'parse_coverage' : '1generator/3parse_coverage',
                'fully_correct' : '1generator/3fully_correct', 'soft_entropy' : '1generator/4soft_entropy', 'genclass_acc' : '1generator/5genclass_acc',
                'genclass_entropy' : '1generator/5genclass_entropy', 'score_gen_alt' : '1generator/1score_gen',
                'crossentropy_real' : '4extra/1ce_real', 'crossentropy_gen' : '4extra/2ce_gen', 'crossentropy_genalt' : '4extra/3ce_genalt',
                'wasserstein' : '1critic/1wasserstein', 'logits_real' : '1critic/1logits_real', 'logits_gen' : '1critic/1logits_gen',
                'class_entropy' : '1critic/2class_entropy', 'score_gen_hard' : '1generator/6score_gen_hard', 'logits_gen_hard' : '1generator/6logits_gen_hard',
                'genclass_entropy_hard' : '1generator/5genclass_entropy_hard', 'genclass_logits_hard_mean' : '1generator/5genclass_logits_hard',
                'class_acc_from_buffer' : '4extra/6class_acc_from_buffer', 'class_acc_from_dataset' : '4extra/6class_acc_from_dataset', 'min_logits_gen' : '4extra/7min_logits_gen',
                'class_loss' : '4extra/8class_loss_real'
                }
            for k, v in name_mapping.items():
                if k in metrics:
                    tf.summary.scalar(v, metrics[k])

        proforma_loss = 0 # TODO
        metrics['loss'] = proforma_loss
        if 'soft_entropy' in metrics and metrics['soft_entropy'] == 0:
            del metrics['soft_entropy']

        self.epoch_steps += 1
        self.total_steps += 1
        return metrics        


    def analyze_generated(self, tokens):
        """copmute metrics for generated samples

        tokens : list of list of string tokens (n/bs -> sequence length)
        returns dict with scores
        """
        res = {}
        num_analyze = len(tokens)
        hard_entropy, fragments, coverage, fully_correct = 0, 0, 0, 0
        for q in tokens:
            if self.params['problem'] == 'ltl':
                e, f, c = parse_score(''.join(q))
            elif self.params['problem'] == 'math':
                f, c = parse_score_math(q)
                e = 0
            if e is None:
                num_analyze -= 1
                continue
            hard_entropy += e
            fragments += f
            coverage += c
            if f == 1 and c == 1.:
                fully_correct += 1
        res['seq_entropy'] = hard_entropy / num_analyze if num_analyze > 0 else None
        res['parse_fragments'] = fragments / num_analyze if num_analyze > 0 else None
        res['parse_coverage'] = coverage / num_analyze if num_analyze > 0 else None
        res['fully_correct'] = fully_correct / num_analyze if num_analyze > 0 else None
        return res


    def test_step(self, data):
        x, y_target = data
        x, _ = x
        batch_size = tf.shape(x)[0]
        res = {}

        # Real (class)
        x_soft, x_mask = self.encode_real(x)
        y_t = tf.squeeze(tf.cast(y_target == 2, tf.float32))
        ??_real = x_soft
        pred_raw = self.critic(??_real, x_mask, training=False)

        if 'class' in self.objectives or self.inherent_class_loss:
            res['class_acc'] = tf.keras.metrics.binary_accuracy(y_t, tf.nn.sigmoid(pred_raw[:, 0]))

        if 'gan' in self.objectives:
            pred_real = tf.nn.sigmoid(pred_raw[:, 1])
            res['score_real'] = tf.reduce_mean(pred_real)
            # Generated
            z, generated_positive_mask = self.input_noise(1, batch_size,)
            ??_gen = self.generator(z, generated_positive_mask, training=True) # lol
            predictions_gen_raw = self.critic(??_gen, generated_positive_mask, training=False)
            predictions_gen = tf.nn.sigmoid(predictions_gen_raw[:, 1])
            res['score_gen'] = tf.reduce_mean(predictions_gen)

            # Analysis
            num_analyze = min(200, batch_size.numpy())
            generated_tokens, ana, generated_hards = self.get_predictions(num_analyze, ??_gen, generated_positive_mask)
            ana.update(self.analyze_generated(generated_tokens))
            res.update(ana)
            if self.test_steps == 0:
                join_str = ' ' if self.params['problem'] == 'math' else ''
                print(f'test: ' + ', '.join([join_str.join(q) for q in generated_tokens[:6]]))

        # "loss"
        if 'class' in self.objectives and not 'gan' in self.objectives:
            res['loss'] = self.class_loss(y_t, pred_raw[:, 0])
        elif 'gan' in self.objectives:
            res['loss'] = - tf.reduce_mean(tf.math.log(pred_real)) - tf.reduce_mean(tf.math.log(1 - predictions_gen))

        self.test_steps += 1
        with self.tb_writer.as_default(step=self.total_steps): #pylint:disable=not-context-manager
            if 'class' in self.objectives or self.inherent_class_loss:
                tf.summary.scalar('1critic/2class_acc_val', res['class_acc'])
        return res


    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_steps = 0
        self.test_steps = 0
        self.epoch = epoch


    def on_epoch_end(self, epoch, logs=None):
        print(self.total_steps, 'steps so far')
        for k, v in self.warnings.items():
            print('Warning:', v)
        self.warnings = {}


    def get_predictions(self, num, ??_gen, generated_positive_mask):
        """arg-max generated data and decode"""
        ana = {}
        generated_hards, soft_entropy = proc_logits(??_gen, generated_positive_mask, sample=True, tau=0, calc_entropy=True)
        ana['soft_entropy'] = soft_entropy
        generated_tokens = self.hard_decode(generated_hards, generated_positive_mask[:num])
        return generated_tokens, ana, generated_hards



def proc_logits(logits, mask=None, normalize=False, sample=False, tau=1, reduce=True, calc_entropy=False):
    """versatile functions for processing one-hot logits"""
    if calc_entropy:
        if normalize:
            softs = tf.nn.softmax(logits)
        else:
            softs = logits
        w = tf.cast(mask, tf.float32)
        ent_per_pos = -tf.reduce_sum(softs * tf.math.log(softs), axis=-1)
        soft_entropy = tf.reduce_mean(tf.reduce_sum(ent_per_pos * w / tf.reduce_sum(w, axis=1, keepdims=True), axis=1))
    if normalize:
        x = tf.nn.softmax(logits / tau)
    else:
        x = logits
    if not sample:
        return (x, soft_entropy) if calc_entropy else x
    if tau != 0:
        raise NotImplementedError
    else: # tau == 0
        res = tf.argmax(x, axis=-1, output_type=tf.dtypes.int32) # int32?
    if not reduce:
        res = tf.one_hot(res, tf.shape(logits)[-1])
    return (res, soft_entropy) if calc_entropy else res


def assert_finite(values, message, info=None):
    """Check if a tensor contains inf or nan, fail in that case"""
    if not all([tf.reduce_all(tf.math.is_finite(q)) for q in values]):
        print('--------------------- FINITE ASSERTION FAILED :', message)
        if info is not None:
            for k, v in info.items():
                print(k, v)
        assert False