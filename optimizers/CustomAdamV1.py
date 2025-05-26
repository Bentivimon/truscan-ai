import tensorflow as tf
from tensorflow.keras.optimizers import Adam

class CustomAdamOptimizerForDeepFake(Adam):
    def __init__(self, learning_rate=0.001, clip_norm=1.0, debug=False, **kwargs):
        super(CustomAdamOptimizerForDeepFake, self).__init__(learning_rate=learning_rate, **kwargs)
        self.clip_norm = clip_norm
        self.debug = debug  # Прапорець для включення логування

    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
        grads_and_vars_list = list(grads_and_vars)

        if not grads_and_vars_list:
            raise ValueError("`grads_and_vars` is empty or None, cannot apply gradients.")

        # Clip gradients
        clipped_grads_and_vars = [(tf.clip_by_norm(grad, self.clip_norm), var) for grad, var in grads_and_vars_list]

        # Логування тільки якщо debug=True
        if self.debug:
            for grad, var in clipped_grads_and_vars:
                print(f"Var: {var.name}, Clipped Grad: {grad}")

        return super(CustomAdamOptimizerForDeepFake, self).apply_gradients(clipped_grads_and_vars)
