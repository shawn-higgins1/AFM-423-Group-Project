	��a��85@��a��85@!��a��85@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��a��85@>x�҆�2@1����u��?AM�St$�?I߉Y/�R@*	�����YI@2U
Iterator::Model::ParallelMapV2�St$���?!<KQ�^@@)�St$���?1<KQ�^@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�(��0�?!�k���B8@)�0�*�?1�S���P4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateZd;�O��?!u��ô�6@)�q����?1O��N��.@:Preprocessing2F
Iterator::Modelg��j+��?!;�;�G@)_�Q�{?1��!��*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vOn?!7Ir�0@)���_vOn?17Ir�0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�#�����?!�N��N�J@){�G�zd?1�v/FO�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!۽=�@)����Mb`?1۽=�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�������?!ST���8@)����MbP?1۽=��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�Y�ȉ�X@QH'S�;�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	>x�҆�2@>x�҆�2@!>x�҆�2@      ��!       "	����u��?����u��?!����u��?*      ��!       2	M�St$�?M�St$�?!M�St$�?:	߉Y/�R@߉Y/�R@!߉Y/�R@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�Y�ȉ�X@yH'S�;�?