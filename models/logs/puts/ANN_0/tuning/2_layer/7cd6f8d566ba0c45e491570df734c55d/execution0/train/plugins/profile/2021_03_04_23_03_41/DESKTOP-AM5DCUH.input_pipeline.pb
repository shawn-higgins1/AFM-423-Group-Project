	����O_4@����O_4@!����O_4@	�N&���?�N&���?!�N&���?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6����O_4@�d�z��1@1��%���?A���Mb�?I&��|�@Y
��a��?*�����YH@)       =2U
Iterator::Model::ParallelMapV2���Q��?!y�ڠ�>@)���Q��?1y�ڠ�>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��<,Ԋ?!��[>�:@)�I+��?16�BW�6@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapM�St$�?!�j�
47@)��H�}}?1"@���-@:Preprocessing2F
Iterator::Model䃞ͪϕ?!ÔSw[�E@) �o_�y?1���h�)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice	�^)�p?!�+K�x� @)	�^)�p?1�+K�x� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�?�߾�?!=k���!L@)�~j�t�h?1ƊH�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!��e�?@)�J�4a?1��e�?@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�N&���?I-�Fo�X@Q`~,1��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�d�z��1@�d�z��1@!�d�z��1@      ��!       "	��%���?��%���?!��%���?*      ��!       2	���Mb�?���Mb�?!���Mb�?:	&��|�@&��|�@!&��|�@B      ��!       J	
��a��?
��a��?!
��a��?R      ��!       Z	
��a��?
��a��?!
��a��?b      ��!       JGPUY�N&���?b q-�Fo�X@y`~,1��?