	�����6@�����6@!�����6@	���f�0�?���f�0�?!���f�0�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�����6@G<�͌v3@1J)�����?A��j+���?I�rK��@YIh˹Wu?*	�����LK@2U
Iterator::Model::ParallelMapV2?�ܵ�|�?!̷|˷|=@)?�ܵ�|�?1̷|˷|=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�
F%u�?!)�B)�B7@)�g��s��?1��i��i3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���QI�?!�0�0:@)��_�L�?1�0�03@:Preprocessing2F
Iterator::Model��ͪ�Ֆ?!��k��kD@)a��+ey?1_�^�&@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!$I�$I�@)�q����o?1$I�$I�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipaTR'���?!C)�B)�M@)_�Q�k?1��舎�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!O��N��@)�J�4a?1O��N��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��ǘ���?!������=@)ŏ1w-!_?1l��k��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���f�0�?I�o9IҤX@Q��	�j�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	G<�͌v3@G<�͌v3@!G<�͌v3@      ��!       "	J)�����?J)�����?!J)�����?*      ��!       2	��j+���?��j+���?!��j+���?:	�rK��@�rK��@!�rK��@B      ��!       J	Ih˹Wu?Ih˹Wu?!Ih˹Wu?R      ��!       Z	Ih˹Wu?Ih˹Wu?!Ih˹Wu?b      ��!       JGPUY���f�0�?b q�o9IҤX@y��	�j�?