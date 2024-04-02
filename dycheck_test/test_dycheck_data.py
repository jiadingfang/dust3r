import os
import warnings
import os.path as osp
import gin
import tensorflow as tf
from absl import app, flags, logging

from dycheck import core, geometry

flags.DEFINE_multi_string(
    "gin_configs",
    "dycheck_test/iphone_dataset.gin",
    "Gin config files.",
)
flags.DEFINE_multi_string("gin_bindings", None, "Gin parameter bindings.")
FLAGS = flags.FLAGS

DATA_ROOT = "/data/dycheck"
FPS = 20

def prepare_dataset():
    tf.config.experimental.set_visible_devices([], "GPU")

    core.parse_config_files_and_bindings(
        config_files=FLAGS.gin_configs,
        bindings=(FLAGS.gin_bindings or [])
        + [
            "Config.engine_cls=None",
            "Config.model_cls=None",
            "SEQUENCE='paper-windmill'",
            f"iPhoneParser.data_root='{DATA_ROOT}'",
            f"iPhoneDatasetFromAllFrames.split='train'",
            f"iPhoneDatasetFromAllFrames.training=False",
            f"iPhoneDatasetFromAllFrames.bkgd_points_batch_size=0",
        ],
        skip_unknown=True,
    )

    config_str = gin.config_str()
    logging.info(f"*** Configuration:\n{config_str}")

    config = core.Config()

    dataset = config.dataset_cls()
    return dataset

def main(_):
    dataset = prepare_dataset()
    print('Dataset length:', len(dataset))

    # Example usage of the dataset
    idx = 0
    sample = dataset[idx]
    camera = dataset.cameras[idx]
    intrin = camera.intrin
    extrin = camera.extrin

    print('Sample keys:', sample.keys())  # Expected output: dict_keys(['rgb', 'mask', 'rays', 'depth'])
    print('Intrin:', intrin)
    print('Extrin:', extrin)

if __name__ == "__main__":
    app.run(main)
