	c��	�8@c��	�8@!c��	�8@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-c��	�8@��/Ie
5@1!W�Y��?A�H.�!��?I���vh�@*	fffff&K@2U
Iterator::Model::ParallelMapV2X�5�;N�?!%/�d�?@)X�5�;N�?1%/�d�?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatS�!�uq�?!�D�~�8@)�+e�X�?1��2��4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�������?!u��/7@)/n���?1am��40@:Preprocessing2F
Iterator::ModeltF��_�?!]e�%P�E@)lxz�,C|?1*7�j)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vOn?!R,@�A@)���_vOn?1R,@�A@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�Zd;�?!��HگL@)_�Q�k?1"�O�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!�R�0Gw@)����Mb`?1�R�0Gw@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�?�߾�?!�a��:9@)a2U0*�S?1�1����@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�-�X@Q�DϜ��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��/Ie
5@��/Ie
5@!��/Ie
5@      ��!       "	!W�Y��?!W�Y��?!!W�Y��?*      ��!       2	�H.�!��?�H.�!��?!�H.�!��?:	���vh�@���vh�@!���vh�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�-�X@y�DϜ��?