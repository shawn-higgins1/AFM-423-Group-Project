	[A�+�6@[A�+�6@![A�+�6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-[A�+�6@�x�0Dz4@1��C�.�?A+��Χ?I��Bb@*	������F@2U
Iterator::Model::ParallelMapV2�~j�t��?!^�(�u�:@)�~j�t��?1^�(�u�:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-C��6�?!�Ź�Q<@)�0�*�?1�|٠�6@:Preprocessing2F
Iterator::Model�l����?!o?$�vD@)9��v��z?1O-����,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�j+��݃?!�\�(�u5@)�����w?1��ZX�)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!/4��A!@)�q����o?1/4��A!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipz6�>W�?!�����M@)�~j�t�h?1^�(�u�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{�G�zd?!x�!��@){�G�zd?1x�!��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap46<�R�?!8�1�s8@)a2U0*�S?1��S+=@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIL�T�{�X@Q�Ǫ��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�x�0Dz4@�x�0Dz4@!�x�0Dz4@      ��!       "	��C�.�?��C�.�?!��C�.�?*      ��!       2	+��Χ?+��Χ?!+��Χ?:	��Bb@��Bb@!��Bb@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qL�T�{�X@y�Ǫ��?