	����+�3@����+�3@!����+�3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-����+�3@���u2@1��g�,�?A���?I��a�?*	gffff�E@2U
Iterator::Model::ParallelMapV2�(��0�?!Q�B�
<@)�(��0�?1Q�B�
<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeattF��_�?!��J@+;@)n���?1�����_6@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��_�L�?!�{�7@)S�!�uq{?1�����.@:Preprocessing2F
Iterator::Model���&�?!��5�XE@)-C��6z?1��QNG9-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice���_vOn?!RNG9� @)���_vOn?1RNG9� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipݵ�|г�?!r:��(�L@)Ǻ���f?1"��d�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!ٲe˖-@)�J�4a?1ٲe˖-@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�s��'�X@Q�#���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���u2@���u2@!���u2@      ��!       "	��g�,�?��g�,�?!��g�,�?*      ��!       2	���?���?!���?:	��a�?��a�?!��a�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�s��'�X@y�#���?