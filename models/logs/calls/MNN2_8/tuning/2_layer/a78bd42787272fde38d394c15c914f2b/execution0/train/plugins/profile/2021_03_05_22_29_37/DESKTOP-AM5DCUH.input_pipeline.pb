	�8�� �6@�8�� �6@!�8�� �6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�8�� �6@��L0�s4@1���(�?AM�St$�?I�شR�@*	������G@2U
Iterator::Model::ParallelMapV2��H�}�?!��sHM0>@)��H�}�?1��sHM0>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat������?!S)ϖ�Q8@)n���?1���퉋4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�+e�X�?!_�.=�7@)�<,Ԛ�}?1������.@:Preprocessing2F
Iterator::Model�0�*�?!����E@)a��+ey?1#aG7��)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	�^)�p?!<Ь�0!@)	�^)�p?1<Ь�0!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��6��?!`1�hL@)�~j�t�h?1<Eg@(@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!��sHM0@)��H�}]?1��sHM0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�������?!hVk�4:@)/n��R?1�e*��r@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIo���1�X@Q5�����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��L0�s4@��L0�s4@!��L0�s4@      ��!       "	���(�?���(�?!���(�?*      ��!       2	M�St$�?M�St$�?!M�St$�?:	�شR�@�شR�@!�شR�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qo���1�X@y5�����?