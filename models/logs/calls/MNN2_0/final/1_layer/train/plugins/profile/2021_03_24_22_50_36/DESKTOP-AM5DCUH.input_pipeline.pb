	�(�1w;@�(�1w;@!�(�1w;@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�(�1w;@�7>[�8@1������?A��Y�rL�?I'����	@*	�����LL@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenatew-!�l�?!���q�XC@)Dio��ɔ?1.7y��A@:Preprocessing2U
Iterator::Model::ParallelMapV2���H�?!�����<@)���H�?1�����<@:Preprocessing2F
Iterator::Model��0�*�?!��;�D@)� �	�?1�z�^5+@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�HP�x?!M�tm�%@)����Mbp?1H#ƿD@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipL7�A`�?!b�K��&M@)F%u�k?1�c�iQR@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!�~�)��@)�J�4a?1�~�)��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��0�*�?!��;�D@)_�Q�[?10��<@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��H�}M?!�l�q�?)��H�}M?1�l�q�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!�q���?)Ǻ���F?1�q���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��)�X@Q��9�u8�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�7>[�8@�7>[�8@!�7>[�8@      ��!       "	������?������?!������?*      ��!       2	��Y�rL�?��Y�rL�?!��Y�rL�?:	'����	@'����	@!'����	@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��)�X@y��9�u8�?