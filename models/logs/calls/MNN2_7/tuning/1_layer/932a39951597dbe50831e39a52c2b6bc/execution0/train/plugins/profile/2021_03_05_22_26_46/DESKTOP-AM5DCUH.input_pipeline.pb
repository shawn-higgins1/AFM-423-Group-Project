	���A_"5@���A_"5@!���A_"5@	����~c�?����~c�?!����~c�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6���A_"5@��G���2@1��:ǀ��?Ao�ŏ1�?I��};�H @Y�fh<q?*	ffffffP@2U
Iterator::Model::ParallelMapV2�0�*��?!dp>�>@)�0�*��?1dp>�>@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�0�*��?!dp>�>@)����Mb�?1dp>�c8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�?�߾�?!�v%jW�4@)A��ǘ��?1,Q���0@:Preprocessing2F
Iterator::Model��6��?!��|ΧD@)y�&1�|?1W�v%jW%@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	�^)�p?!      @)	�^)�p?1      @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��ʡE��?!8��1XM@)_�Q�k?1��+Q�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_�Le?!�+Q��@)��_�Le?1�+Q��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap'�����?!����O@@)��_�LU?1�+Q���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9����~c�?IK��d�X@Q�h��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��G���2@��G���2@!��G���2@      ��!       "	��:ǀ��?��:ǀ��?!��:ǀ��?*      ��!       2	o�ŏ1�?o�ŏ1�?!o�ŏ1�?:	��};�H @��};�H @!��};�H @B      ��!       J	�fh<q?�fh<q?!�fh<q?R      ��!       Z	�fh<q?�fh<q?!�fh<q?b      ��!       JGPUY����~c�?b qK��d�X@y�h��?