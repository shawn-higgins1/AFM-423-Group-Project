	�u�+.�5@�u�+.�5@!�u�+.�5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�u�+.�5@#�GG43@1�sa���?AnN%@�?I\Ɏ�@<@*	33333sK@2U
Iterator::Model::ParallelMapV2/n���?!��x�u@@)/n���?1��x�u@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatlxz�,C�?!�f�"9@)�~j�t��?1'<�ߠ�5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateg��j+��?!�:ڼO5@)�q����?1f�"Qj,@:Preprocessing2F
Iterator::Model��+e�?!�y��!F@)S�!�uq{?1�Ϛ�sh(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!f�"Qj@)�q����o?1f�"Qj@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipK�=�U�?!K�`m�K@)y�&1�l?1.�˯;�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!���:
@)��H�}]?1���:
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�]K�=�?!E�gM�98@)-C��6Z?1��i��P@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIT-�GS�X@Q�t.k�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	#�GG43@#�GG43@!#�GG43@      ��!       "	�sa���?�sa���?!�sa���?*      ��!       2	nN%@�?nN%@�?!nN%@�?:	\Ɏ�@<@\Ɏ�@<@!\Ɏ�@<@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qT-�GS�X@y�t.k�?