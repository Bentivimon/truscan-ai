import tensorflow as tf
import numpy as np


class CustomAdamV2(tf.keras.optimizers.Adam):
    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7, weight_decay=0.0, name="CustomAdam",
                 **kwargs):
        super().__init__(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, name=name,
                         **kwargs)
        self.weight_decay = weight_decay

    def _resource_apply_dense(self, grad, var, apply_state=None):
        if self.weight_decay > 0:
            grad += self.weight_decay * var  # Додаємо weight decay

        return super()._resource_apply_dense(grad, var, apply_state)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        if self.weight_decay > 0:
            grad += self.weight_decay * tf.gather(var, indices)  # Аналогічно для розріджених градієнтів

        return super()._resource_apply_sparse(grad, var, indices, apply_state)


# optimizer = CustomAdam(learning_rate=3e-4, weight_decay=1e-4)

class CustomAdamV3(tf.keras.optimizers.Adam):
    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7, weight_decay=0.0, grad_clip=1.0,
                 **kwargs):
        super().__init__(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, **kwargs)
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip  # Обмеження градієнтів

    def _resource_apply_dense(self, grad, var, apply_state=None):
        norm = tf.norm(grad)
        if norm > self.grad_clip:
            grad = grad * (self.grad_clip / norm)  # Нормалізація градієнтів

        if self.weight_decay > 0:
            grad += self.weight_decay * var  # Додаємо weight decay

        return super()._resource_apply_dense(grad, var, apply_state)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        if self.weight_decay > 0:
            grad += self.weight_decay * tf.gather(var, indices)  # Аналогічно для розріджених градієнтів

        return super()._resource_apply_sparse(grad, var, indices, apply_state)


# optimizer = CustomAdamV3(learning_rate=1e-3, grad_clip=1.0)

class CustomAdamV4(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=1e-3, beta_scale=0.05, lambda_weight=0.5, switch_epoch=5, **kwargs):
        super().__init__(**kwargs)

        # Власні параметри
        self.learning_rate = learning_rate
        self.beta_scale = beta_scale
        self.lambda_weight = lambda_weight
        self.switch_epoch = switch_epoch

    def _create_slots(self, var):
        """Створення слотів для m і v для кожної змінної"""
        self.add_slot(var, "m")
        self.add_slot(var, "v")

    def _decayed_lr(self, var_dtype):
        """Метод для динамічного зменшення learning rate"""
        lr_t = self.learning_rate
        # Перевірка, чи потрібно змінювати learning rate
        lr_t = tf.cond(
            self.iterations >= self.switch_epoch,
            lambda: lr_t * self.lambda_weight,  # Зменшуємо learning rate
            lambda: lr_t  # Якщо не досягнуто switch_epoch, повертаємо поточний learning rate
        )
        return lr_t

    def _apply_updates(self, grad, var):
        """Головна логіка оновлення ваг"""
        if not isinstance(var, tf.Variable):
            var = tf.Variable(var)

        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)  # Отримуємо поточний learning rate

        # Ініціалізуємо слоти для m і v, якщо вони ще не існують
        if "m" not in self.get_slot_names(var):
            self.add_slot(var, "m")
        if "v" not in self.get_slot_names(var):
            self.add_slot(var, "v")

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        curvature = tf.abs(v / (tf.square(m) + self.epsilon))

        new_beta_1 = tf.clip_by_value(self.beta_1 + self.beta_scale * curvature, 0.85, 0.99)
        new_beta_2 = tf.clip_by_value(self.beta_2 + self.beta_scale * curvature, 0.995, 0.9999)

        m_t = new_beta_1 * m + (1.0 - new_beta_1) * grad
        v_t = new_beta_2 * v + (1.0 - new_beta_2) * tf.square(grad)

        t = tf.cast(self.iterations + 1, var_dtype)
        m_hat = m_t / (1.0 - tf.pow(new_beta_1, t))
        v_hat = v_t / (1.0 - tf.pow(new_beta_2, t))

        var_update = var - lr_t * m_hat / (tf.sqrt(v_hat) + self.epsilon)
        var.assign(var_update)

        m.assign(m_t)
        v.assign(v_t)

    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
        """Метод застосування градієнтів"""
        for grad, var in grads_and_vars:
            self._apply_updates(grad, var)

    def get_config(self):
        """ Серіалізація гіперпараметрів """
        config = super().get_config()
        config.update({
            "sync_period": self.sync_period,
            "slow_step": self.slow_step,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "base_optimizer": tf.keras.optimizers.serialize(self.base_optimizer)
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["base_optimizer"] = tf.keras.optimizers.deserialize(config["base_optimizer"])
        return cls(**config)


# Використання
# base_adam = tf.keras.optimizers.Adam(learning_rate=1e-3)
# optimizer = LookaheadAdam(base_optimizer=base_adam, sync_period=5, slow_step=0.5)

class CustomAdamV5(tf.keras.optimizers.Adam):
    """Гібридний оптимізатор, що комбінує CustomAdamV5 та LAMB"""


def __init__(self, learning_rate=1e-3, beta_scale=0.05, lambda_weight=0.5, switch_epoch=5, name="HybridOptimizer",
             **kwargs):
    super().__init__(name=name, **kwargs)
    self._set_hyper("learning_rate", learning_rate)
    self.beta_scale = beta_scale
    self.lambda_weight = lambda_weight
    self.switch_epoch = switch_epoch


def _create_slots(self, var_list):
    """Створює слоти для моменту m і другого моменту v"""
    for var in var_list:
        self.add_slot(var, "m")
        self.add_slot(var, "v")


def _resource_apply_dense(self, grad, var, apply_state=None):
    """Головна логіка оновлення ваг"""
    var_dtype = var.dtype.base_dtype
    lr_t = self._get_hyper("learning_rate", var_dtype)

    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")

    curvature = tf.abs(v / (tf.square(m) + 1e-7))

    new_beta_1 = tf.clip_by_value(0.9 + self.beta_scale * curvature, 0.85, 0.99)
    new_beta_2 = tf.clip_by_value(0.999 + self.beta_scale * curvature, 0.995, 0.9999)

    m_t = new_beta_1 * m + (1.0 - new_beta_1) * grad
    v_t = new_beta_2 * v + (1.0 - new_beta_2) * tf.square(grad)

    t = tf.cast(self.iterations + 1, var_dtype)
    m_hat = m_t / (1.0 - tf.pow(new_beta_1, t))
    v_hat = v_t / (1.0 - tf.pow(new_beta_2, t))

    r1 = tf.norm(var)
    r2 = tf.norm(m_hat / (tf.sqrt(v_hat) + 1e-7))
    trust_ratio = tf.where(r1 > 0, tf.where(r2 > 0, r1 / r2, 1.0), 1.0)

    var_update = var - lr_t * trust_ratio * m_hat / (tf.sqrt(v_hat) + 1e-7)
    var.assign(var_update)

    m.assign(m_t)
    v.assign(v_t)


def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
    """Метод застосування градієнтів"""
    for grad, var in grads_and_vars:
        self._resource_apply_dense(grad, var)


def get_config(self):
    config = super().get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "beta_scale": self.beta_scale,
        "lambda_weight": self.lambda_weight,
        "switch_epoch": self.switch_epoch
    })
    return config


# optimizer_resnet = CustomAdamV5(learning_rate=3e-4, beta_scale=0.05)

class CustomAdamV6(tf.keras.optimizers.Adam):
    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7, lambda_reg=0.01,
                 name="SecondOrderAdam", **kwargs):
        super().__init__(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, name=name,
                         **kwargs)
        self.lambda_reg = lambda_reg  # коефіцієнт регуляризації другого порядку

    def _resource_apply_dense(self, grad, var, apply_state=None):
        # Обчислення другої похідної (наближено)
        hessian_approx = tf.square(grad)  # Можна замінити на більш точний розрахунок Гессіана
        reg_term = self.lambda_reg * hessian_approx

        # Додаємо регуляризацію другого порядку
        new_grad = grad + reg_term

        return super()._resource_apply_dense(new_grad, var, apply_state)


# Використання
# optimizer = CustomAdamV6(learning_rate=3e-4, lambda_reg=0.05)



class CustomAdamV5(tf.keras.optimizers.Adam):
    """Адам-оптимізатор із динамічною зміною β1/β2 на основі кривизни."""

    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7, beta_scale=0.05, name="CustomAdamV5",
                 **kwargs):
        super().__init__(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, name=name,
                         **kwargs)
        self.beta_scale = beta_scale

    def _resource_apply_dense(self, grad, var, apply_state=None):
        """Оновлення ваг із урахуванням кривизни."""
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        curvature = tf.abs(v / (tf.square(m) + self.epsilon))

        new_beta_1 = tf.clip_by_value(self.beta_1 + self.beta_scale * curvature, 0.85, 0.99)
        new_beta_2 = tf.clip_by_value(self.beta_2 + self.beta_scale * curvature, 0.995, 0.9999)

        m_t = new_beta_1 * m + (1.0 - new_beta_1) * grad
        v_t = new_beta_2 * v + (1.0 - new_beta_2) * tf.square(grad)

        t = tf.cast(self.iterations + 1, var_dtype)
        m_hat = m_t / (1.0 - tf.pow(new_beta_1, t))
        v_hat = v_t / (1.0 - tf.pow(new_beta_2, t))

        var_update = var - lr_t * m_hat / (tf.sqrt(v_hat) + self.epsilon)
        var.assign(var_update)

        m.assign(m_t)
        v.assign(v_t)

    def get_config(self):
        config = super().get_config()
        config.update({"beta_scale": self.beta_scale})
        return config


class HybridOptimizer(tf.keras.optimizers.Optimizer):
    """Комбінований оптимізатор: CustomAdamV5 + LAMB"""

    def __init__(self, learning_rate=3e-4, beta_scale=0.05, lambda_weight=0.5, switch_epoch=5, name="HybridOptimizer"):
        # Ініціалізація через базовий клас і встановлення гіперпараметрів
        super().__init__(learning_rate=learning_rate)
        self.beta_scale = beta_scale
        self.lambda_weight = lambda_weight
        self.switch_epoch = switch_epoch

    def build(self, var_list):
        print(f"Building optimizer slots for variables: {[var.name for var in var_list]}")
        for var in var_list:
            self.add_variable_from_reference(var, "m")
            self.add_variable_from_reference(var, "v")

    def update_step(self, grad, var):
        """Головна логіка оновлення змінних"""
        var_dtype = var.dtype.base_dtype
        lr_t = self._get_hyper("learning_rate", var_dtype)

        m = self.get_variable_reference(var, "m")
        v = self.get_variable_reference(var, "v")

        # Динамічна зміна коефіцієнтів beta
        curvature = tf.abs(v / (tf.square(m) + 1e-7))
        beta_1 = tf.clip_by_value(0.9 + self.beta_scale * curvature, 0.85, 0.99)
        beta_2 = tf.clip_by_value(0.999 + self.beta_scale * curvature, 0.995, 0.9999)

        # Оновлення моментів
        m.assign(beta_1 * m + (1.0 - beta_1) * grad)
        v.assign(beta_2 * v + (1.0 - beta_2) * tf.square(grad))

        # Розрахунок "trust ratio"
        m_hat = m / (1.0 - tf.pow(beta_1, tf.cast(self.iterations + 1, var_dtype)))
        v_hat = v / (1.0 - tf.pow(beta_2, tf.cast(self.iterations + 1, var_dtype)))
        trust_ratio = tf.norm(var) / (tf.norm(m_hat) + 1e-6)
        trust_ratio = tf.clip_by_value(trust_ratio, 0.1, 10.0)

        # Оновлення змінних
        var.assign_sub(lr_t * trust_ratio * m_hat / (tf.sqrt(v_hat) + 1e-7))

    def get_config(self):
        """Серіалізація гіперпараметрів"""
        return {
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "beta_scale": self.beta_scale,
            "lambda_weight": self.lambda_weight,
            "switch_epoch": self.switch_epoch
        }

class HybridOptimizerSimple(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=3e-4, name="HybridOptimizerMinimal"):
        super().__init__(name=name, learning_rate=learning_rate)  # Передаємо лише назву

    def build(self, var_list):
        # Мінімальне створення слотів
        for var in var_list:
            self.add_variable_from_reference(var, "m")

    def update_step(self, grad, var):
        # Мінімальна операція оновлення
        var.assign_sub(self._get_hyper("learning_rate") * grad)

class AdamM_Hessian(tf.keras.optimizers.Optimizer):
    """Оптимізатор з Гессіаном для врахування локальної кривизни"""

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, lambda_reg=0.01, name="AdamM_Hessian"):
        super().__init__(name=name)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.lambda_reg = lambda_reg

    def build(self, var_list):
        """Створення слотів для моментів і Гессе"""
        for var in var_list:
            self.add_variable_from_reference(var, "m")
            self.add_variable_from_reference(var, "v")
            self.add_variable_from_reference(var, "hessian")

    def update_step(self, grad, var):
        """Оновлення параметрів"""
        var_dtype = var.dtype.base_dtype
        lr_t = tf.convert_to_tensor(self.learning_rate, dtype=var_dtype)

        m = self.get_variable_reference(var, "m")
        v = self.get_variable_reference(var, "v")
        hessian = self.get_variable_reference(var, "hessian")

        # Оновлення моментів
        m.assign(self.beta_1 * m + (1 - self.beta_1) * grad)
        v.assign(self.beta_2 * v + (1 - self.beta_2) * tf.square(grad))

        # Оновлення Гессе (наближене)
        hessian.assign(tf.square(grad) + self.lambda_reg)

        # Розрахунок оновлення ваг
        m_hat = m / (1 - tf.pow(self.beta_1, tf.cast(self.iterations + 1, var_dtype)))
        v_hat = v / (1 - tf.pow(self.beta_2, tf.cast(self.iterations + 1, var_dtype)))

        var_update = lr_t * m_hat / (tf.sqrt(v_hat) + tf.sqrt(hessian) + self.epsilon)
        var.assign_sub(var_update)

    def get_config(self):
        return {
            "learning_rate": self.learning_rate,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "epsilon": self.epsilon,
            "lambda_reg": self.lambda_reg,
        }
