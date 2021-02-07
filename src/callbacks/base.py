'''
Callback inspritations from PyTorch Lightning - https://github.com/PyTorchLightning/PyTorch-Lightning
and https://github.com/devforfu/pytorch_playground/blob/master/loop.ipynb
'''

import abc


class Callback(abc.ABC):
    def setup(self, **kwargs):
        """Called before the training procedure"""
        pass

    def teardown(self, **kwargs):
        """Called after training procedure"""
        pass

    def on_epoch_start(self, **kwargs):
        """Called when epoch begins"""
        pass

    def on_epoch_end(self, **kwargs):
        """Called when epoch terminates"""
        pass

    def on_train_batch_start(self, **kwargs):
        """Called when training step begins"""
        pass

    def on_train_batch_end(self, **kwargs):
        """Called when training step ends"""
        pass

    def on_validation_batch_start(self, **kwargs):
        """Called when validation step begins"""
        pass

    def on_validation_batch_end(self, **kwargs):
        """Called when validation step ends"""
        pass

    def on_test_batch_start(self, **kwargs):
        """Called when test batch begins"""
        pass

    def on_test_batch_end(self, **kwargs):
        """Called when test batch ends"""
        pass

    def on_train_start(self, **kwargs):
        """Called when training loop begins"""
        pass

    def on_train_end(self, **kwargs):
        """Called when training loop ends"""
        pass

    def on_validation_start(self, **kwargs):
        """Called when validation loop begins"""
        pass

    def on_validation_end(self, **kwargs):
        """Called when validation loop ends"""
        pass

    def on_test_start(self, **kwargs):
        """Called when test loop begins"""
        pass

    def on_test_end(self, **kwargs):
        """Called when test loop ends"""
        pass


class CallbackList(Callback):

    def __init__(self, callbacks):
        self.callbacks = callbacks

    def setup(self, **kwargs):
        """Called before the training procedure"""
        for callback in self.callbacks:
            callback.setup(**kwargs)

    def teardown(self, **kwargs):
        """Called after training procedure"""
        for callback in self.callbacks:
            callback.teardown(**kwargs)

    def on_epoch_start(self, **kwargs):
        """Called when epoch begins"""
        for callback in self.callbacks:
            callback.on_epoch_start(**kwargs)

    def on_epoch_end(self, **kwargs):
        """Called when epoch terminates"""
        for callback in self.callbacks:
            callback.on_epoch_end(**kwargs)

    def on_train_batch_start(self, **kwargs):
        """Called when training step begins"""
        for callback in self.callbacks:
            callback.on_train_batch_start(**kwargs)

    def on_train_batch_end(self, **kwargs):
        """Called when training step ends"""
        for callback in self.callbacks:
            callback.on_train_batch_end(**kwargs)

    def on_validation_batch_start(self, **kwargs):
        """Called when validation step begins"""
        for callback in self.callbacks:
            callback.on_validation_batch_start(**kwargs)

    def on_validation_batch_end(self, **kwargs):
        """Called when validation step ends"""
        for callback in self.callbacks:
            callback.on_validation_batch_end(**kwargs)

    def on_test_batch_start(self, **kwargs):
        """Called when test batch begins"""
        for callback in self.callbacks:
            callback.on_test_batch_start(**kwargs)

    def on_test_batch_end(self, **kwargs):
        """Called when test batch ends"""
        for callback in self.callbacks:
            callback.on_test_batch_end(**kwargs)

    def on_train_start(self, **kwargs):
        """Called when training loop begins"""
        for callback in self.callbacks:
            callback.on_train_start(**kwargs)

    def on_train_end(self, **kwargs):
        """Called when training loop ends"""
        for callback in self.callbacks:
            callback.on_train_end(**kwargs)

    def on_validation_start(self, **kwargs):
        """Called when validation loop begins"""
        for callback in self.callbacks:
            callback.on_validation_start(**kwargs)

    def on_validation_end(self, **kwargs):
        """Called when validation loop ends"""
        for callback in self.callbacks:
            callback.on_validation_end(**kwargs)

    def on_test_start(self, **kwargs):
        """Called when test loop begins"""
        for callback in self.callbacks:
            callback.on_test_start(**kwargs)

    def on_test_end(self, **kwargs):
        """Called when test loop ends"""
        for callback in self.callbacks:
            callback.on_test_end(**kwargs)
