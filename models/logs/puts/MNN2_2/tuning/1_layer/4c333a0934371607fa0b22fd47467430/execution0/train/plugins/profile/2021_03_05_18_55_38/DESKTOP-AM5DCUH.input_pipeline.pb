	�X5s�5@�X5s�5@!�X5s�5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�X5s�5@v�1<j3@11ҋ��*�?A�,C��?I\��J�@*	     �F@2U
Iterator::Model::ParallelMapV2�HP��?!-�-�:@)�HP��?1-�-�:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatZd;�O��?!eTFeTF9@)��~j�t�?1��4@:Preprocessing2F
Iterator::Model�ݓ��Z�?!O��N��D@)S�!�uq{?16Ws5Ws-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate/�$��?!;�;�7@)9��v��z?1$I�$I�,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice����Mbp?!R�Q�!@)����Mbp?1R�Q�!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�]K�=�?!�;�;M@)F%u�k?1.�-�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!R�Q�@)����Mb`?1R�Q�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��0�*�?!����9@)��_�LU?1�m۶m�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�#=��X@Q-�p�C�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	v�1<j3@v�1<j3@!v�1<j3@      ��!       "	1ҋ��*�?1ҋ��*�?!1ҋ��*�?*      ��!       2	�,C��?�,C��?!�,C��?:	\��J�@\��J�@!\��J�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�#=��X@y-�p�C�?