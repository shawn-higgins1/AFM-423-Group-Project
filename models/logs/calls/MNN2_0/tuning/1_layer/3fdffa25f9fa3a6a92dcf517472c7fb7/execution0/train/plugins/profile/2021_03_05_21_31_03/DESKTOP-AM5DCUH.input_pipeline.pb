	^-w�6@^-w�6@!^-w�6@	X1����?X1����?!X1����?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6^-w�6@i:;y4@1�q����?AM�St$�?IΪ��V��?Y�s(CUL�?*	�����LI@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate���&�?!7��MozB@)������?1�9��@@:Preprocessing2U
Iterator::Model::ParallelMapV2vq�-�?!�?�9?@)vq�-�?1�?�9?@:Preprocessing2F
Iterator::Model�I+��?!����7�E@)a��+ey?1������(@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��0�*x?!����Q'@)����Mbp?1�C��ܞ@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���QI�?!d!Y�BL@)�����g?1���R��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�3[
@)ŏ1w-!_?1�3[
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��ׁsF�?!Y�B��C@)/n��R?1���,d@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��H�}M?!k#a `u�?)��H�}M?1k#a `u�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!��5g"�?)Ǻ���F?1��5g"�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 90.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9X1����?I9	4&�X@Qy�e+���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	i:;y4@i:;y4@!i:;y4@      ��!       "	�q����?�q����?!�q����?*      ��!       2	M�St$�?M�St$�?!M�St$�?:	Ϊ��V��?Ϊ��V��?!Ϊ��V��?B      ��!       J	�s(CUL�?�s(CUL�?!�s(CUL�?R      ��!       Z	�s(CUL�?�s(CUL�?!�s(CUL�?b      ��!       JGPUYX1����?b q9	4&�X@yy�e+���?