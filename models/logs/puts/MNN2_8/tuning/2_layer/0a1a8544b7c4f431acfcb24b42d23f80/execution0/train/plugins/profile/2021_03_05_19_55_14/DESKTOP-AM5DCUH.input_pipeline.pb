	r3܀ϋ7@r3܀ϋ7@!r3܀ϋ7@	<_��̓?<_��̓?!<_��̓?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6r3܀ϋ7@*s��v4@1��/�$�?A
ףp=
�?I�TO�@Y�9��*�r?*	����̌G@2U
Iterator::Model::ParallelMapV2-C��6�?!"��-;@)-C��6�?1"��-;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��<,Ԋ?!vJa{+�;@)46<�R�?1	��j$7@:Preprocessing2F
Iterator::Model8��d�`�?!�P�d E@)�ZӼ�}?1�'!�&.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��ׁsF�?!�ƾG�5@)-C��6z?1"��-+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey�&1�l?!u��W�@)y�&1�l?1u��W�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip_�Qڛ?!�r���L@)-C��6j?1"��-@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!�����@)/n��b?1�����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap46<�R�?!	��j$7@)����MbP?1�H2� @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 86.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9<_��̓?IOs3��X@QТ'�3�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	*s��v4@*s��v4@!*s��v4@      ��!       "	��/�$�?��/�$�?!��/�$�?*      ��!       2	
ףp=
�?
ףp=
�?!
ףp=
�?:	�TO�@�TO�@!�TO�@B      ��!       J	�9��*�r?�9��*�r?!�9��*�r?R      ��!       Z	�9��*�r?�9��*�r?!�9��*�r?b      ��!       JGPUY<_��̓?b qOs3��X@yТ'�3�?