	3�ۃ�6@3�ۃ�6@!3�ۃ�6@	�n}�?�n}�?!�n}�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails63�ۃ�6@QlMK�3@1�?Pn���?A�=yX��?I�W\�@Y�v�$j?*	fffff&K@2U
Iterator::Model::ParallelMapV2	�^)ː?!��_+�3>@)	�^)ː?1��_+�3>@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate2�%䃎?!�D�~�p;@)A��ǘ��?1=�0&q4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�(��0�?!mǇ �6@)�0�*�?1�u:��2@:Preprocessing2F
Iterator::Modelg��j+��?!U4O��E@)y�&1�|?11h}J^�)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!a��:�@)ŏ1w-!o?1a��:�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��?��?!�˰W�sL@)��_�Le?1��)y!'@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!�R�0Gw@)����Mb`?1�R�0Gw@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap����Mb�?!�R�0Gw=@)/n��R?1am��4 @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�n}�?I����k�X@Q�k�Gq��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	QlMK�3@QlMK�3@!QlMK�3@      ��!       "	�?Pn���?�?Pn���?!�?Pn���?*      ��!       2	�=yX��?�=yX��?!�=yX��?:	�W\�@�W\�@!�W\�@B      ��!       J	�v�$j?�v�$j?!�v�$j?R      ��!       Z	�v�$j?�v�$j?!�v�$j?b      ��!       JGPUY�n}�?b q����k�X@y�k�Gq��?