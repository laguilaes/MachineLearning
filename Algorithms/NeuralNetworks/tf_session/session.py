def set_seed(seed=42):
    from tensorflow.compat.v1 import ConfigProto, GPUOptions, InteractiveSession
    import random
    import tensorflow as tf
    random.seed(seed)
    tf.random.set_seed(seed)
    # Option1
    # session_conf = tf.compat.v1.ConfigProto(
    #     intra_op_parallelism_threads=1,
    #     inter_op_parallelism_threads=1
    # )
    # sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    # tf.compat.v1.keras.backend.set_session(sess)
    
    #Option 2
    # config = ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = InteractiveSession(config=config)
    
    #Option 3
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    session = InteractiveSession(config=config)
    
set_seed()