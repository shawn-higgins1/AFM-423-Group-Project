	�|x� �5@�|x� �5@!�|x� �5@	0|��h�?0|��h�?!0|��h�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�|x� �5@\���3@1�Y�rL�?A�H�}�?I���� @Y�}i��?*	433333G@2U
Iterator::Model::ParallelMapV2�HP��?!����K:@)�HP��?1����K:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�(��0�?!O#,�4�:@)/�$��?1��{�6@:Preprocessing2F
Iterator::Modela2U0*��?!��=��D@)y�&1�|?15�rO#,.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��_�L�?!�FX�i6@)-C��6z?1a���+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!����=!@)����Mbp?1����=!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip_�Qڛ?!+�4�rOM@)F%u�k?1$,�4�r@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!>����@)��H�}]?1>����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap������?!      9@)a2U0*�S?1��=��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9/|��h�?Iȧ�[�X@Q[v�	i�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	\���3@\���3@!\���3@      ��!       "	�Y�rL�?�Y�rL�?!�Y�rL�?*      ��!       2	�H�}�?�H�}�?!�H�}�?:	���� @���� @!���� @B      ��!       J	�}i��?�}i��?!�}i��?R      ��!       Z	�}i��?�}i��?!�}i��?b      ��!       JGPUY/|��h�?b qȧ�[�X@y[v�	i�?