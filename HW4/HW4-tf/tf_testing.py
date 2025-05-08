import os
import tensorflow as tf

if __name__ == "__main__":
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")

    # Check if GPU is available
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

    # Check CUDA version that TensorFlow was built with
    print(f"TensorFlow built with CUDA: {tf.sysconfig.get_build_info()['cuda_version']}")

    # Try to force CPU/GPU placement to see where the error occurs
    with tf.device('/CPU:0'):
        # Create a simple GRU layer on CPUpip uninstall tensorflow
        cpu_gru = tf.keras.layers.GRU(10, input_shape=(5, 3))
        # Test with random data
        test_input = tf.random.normal((2, 5, 3))
        print("CPU GRU output shape:", cpu_gru(test_input).shape)

    # Optional: check cuDNN version
    try:
        from tensorflow.python.platform import build_info

        print(f"cuDNN version: {build_info.build_info['cudnn_version']}")
    except:
        print("Could not determine cuDNN version from TensorFlow")

    # 3x3x4
    # batch:3, input:3, output: 4
    weights = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], ],
               [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], ],
               [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], ]]

