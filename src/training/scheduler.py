import tensorflow as tf


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, scale=0):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, float)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
        }
        return config

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        # Linear warmup
        warmup_lr = self.base_lr * (step / self.warmup_steps)
        # Cosine decay after warmup
        decay_steps = tf.cast(self.total_steps - self.warmup_steps, tf.float32)
        progress = (step - self.warmup_steps) / decay_steps
        cosine_lr = self.base_lr * 0.5 * (1.0 + tf.cos(3.14159 * progress))
        
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)