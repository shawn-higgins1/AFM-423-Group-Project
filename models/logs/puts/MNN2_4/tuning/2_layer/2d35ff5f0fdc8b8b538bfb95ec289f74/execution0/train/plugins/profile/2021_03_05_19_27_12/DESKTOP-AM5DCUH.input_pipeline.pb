	�(�[Z17@�(�[Z17@!�(�[Z17@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�(�[Z17@jM�S�4@1���B���?A�v���?I��$��@*	gffff�F@2U
Iterator::Model::ParallelMapV2�HP��?!s���6�:@)�HP��?1s���6�:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatZd;�O��?!��̕�9@)��~j�t�?1��-�4@:Preprocessing2F
Iterator::Model�0�*��?!:ܟ�w�E@)vq�-�?1Ct�?1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��~j�t�?!��-�4@)�~j�t�x?12W�ol3*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipp_�Q�?!�#`b�L@)y�&1�l?1�e��S�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey�&1�l?!�e��S�@)y�&1�l?1�e��S�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!!:ܟ�w@)����Mb`?1!:ܟ�w@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap'�����?!�Z9�\7@)a2U0*�S?1(�nY��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�
�.�X@Q@V}z��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	jM�S�4@jM�S�4@!jM�S�4@      ��!       "	���B���?���B���?!���B���?*      ��!       2	�v���?�v���?!�v���?:	��$��@��$��@!��$��@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�
�.�X@y@V}z��?