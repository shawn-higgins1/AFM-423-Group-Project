	��+�,�>@��+�,�>@!��+�,�>@	��Y�,��?��Y�,��?!��Y�,��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��+�,�>@͏����;@1l@��r�?AX9��v�?I~�N�Z/@YT�qs*y?*	      P@2U
Iterator::Model::ParallelMapV2�o_��?!yCސ7�9@)�o_��?1yCސ7�9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatr�����?!�w�q;@)���Q��?1��}A7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate�q����?!����/8@){�G�z�?1���/@:Preprocessing2F
Iterator::Model�(��0�?!�qG�C@)vq�-�?1 ��(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceǺ���v?!W�uE]!@)Ǻ���v?1W�uE]!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipY�8��m�?!<⎸#�N@)��ZӼ�t?1�����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_vOf?!0����@)��_vOf?10����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap������?!�/���:@)�~j�t�X?1��)kʚ@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 90.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��Y�,��?I�'�o�X@Qߘ�T���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	͏����;@͏����;@!͏����;@      ��!       "	l@��r�?l@��r�?!l@��r�?*      ��!       2	X9��v�?X9��v�?!X9��v�?:	~�N�Z/@~�N�Z/@!~�N�Z/@B      ��!       J	T�qs*y?T�qs*y?!T�qs*y?R      ��!       Z	T�qs*y?T�qs*y?!T�qs*y?b      ��!       JGPUY��Y�,��?b q�'�o�X@yߘ�T���?