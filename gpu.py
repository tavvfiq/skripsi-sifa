from typing import (
    Optional,
    List
)
import tensorflow as tf


def list_logical_devices_both_set_control_and_limit(
        memory_limit: Optional[int] = None,
):
    gpus: List[tf.config.PhysicalDevice] = tf.config.list_physical_devices('GPU')
    if not gpus:
        return

    _current: tf.config.PhysicalDevice = None
    try:
        for index, gpu in enumerate(gpus):
            _current = gpu

            # Set memory growth control
            # Currently, memory growth needs to be the same across GPUs
            print(f"setting memory_growth: index:[{index}] gpu:{gpu}")
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')

            # Set memory limit
            # Calling list_logical_devices above should prevent further configuration 
            # here when calling set_logical_device_configuration.
            print(f"setting memory_limit: index:[{index}] gpu:{gpu}")
            tf.config.set_logical_device_configuration(
                device=gpu,
                logical_devices=[
                    tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)
                ]
            )
            logical_gpus = tf.config.list_logical_devices('GPU')
    except RuntimeError as err:
        print(f"Memory growth must be set before GPU [{_current}] have been initialized")
        raise err
    except ValueError as err:
        print(f"Invalid GPU device [{_current}]")
        raise err

def set_memory_limit(
        memory_limit: Optional[int] = None,
):
    gpus: List[tf.config.PhysicalDevice] = tf.config.list_physical_devices('GPU')
    if not gpus:
        return

    _current: tf.config.PhysicalDevice = None
    try:
        for index, gpu in enumerate(gpus):
            _current = gpu

            # Set memory limit
            print(f"setting memory_limit: index:[{index}] gpu:{gpu}")
            tf.config.set_logical_device_configuration(
                device=gpu,
                logical_devices=[
                    tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)
                ]
            )
            logical_gpus = tf.config.list_logical_devices('GPU')
    except RuntimeError as err:
        print(f"Memory growth must be set before GPU [{_current}] have been initialized")
        raise err
    except ValueError as err:
        print(f"Invalid GPU device [{_current}]")
        raise err