	˼Uס7@˼Uס7@!˼Uס7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-˼Uס7@@l��T{4@1>B͐*��?A��&��?I3Mg'�@*�����LG@)       =2U
Iterator::Model::ParallelMapV2�~j�t��?!2NQF�9@)�~j�t��?12NQF�9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��<,Ԋ?!���<@)A��ǘ��?1ՎΤ��7@:Preprocessing2F
Iterator::Model��~j�t�?!�U�bD@)y�&1�|?1�W0��
.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��ZӼ�?!xO�n�5@) �o_�y?1O�n�	+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!�祁�� @)�q����o?1�祁�� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Ziplxz�,C�?!��L��M@)-C��6j?1��x�w@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!�V���*@)����Mb`?1�V���*@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapM�St$�?!�����?8@)/n��R?1��U�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIԎ����X@Q�J\�T�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	@l��T{4@@l��T{4@!@l��T{4@      ��!       "	>B͐*��?>B͐*��?!>B͐*��?*      ��!       2	��&��?��&��?!��&��?:	3Mg'�@3Mg'�@!3Mg'�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qԎ����X@y�J\�T�?