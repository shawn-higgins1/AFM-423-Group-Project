	�}��5@�}��5@!�}��5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�}��5@F{��W3@1��r�4�?A���Mb�?I��d�zh @*	43333�I@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate���&�?!�0�0�0B@)X�5�;N�?1�p�p�p@@:Preprocessing2U
Iterator::Model::ParallelMapV2���QI�?!.�-�-�;@)���QI�?1.�-�-�;@:Preprocessing2F
Iterator::Model��_�L�?!�;�;�;D@)9��v��z?1�J�J�J)@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat_�Q�{?!�u�u�u*@)n��t?1���#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipK�=�U�?!<�;�;�M@)_�Q�k?1�u�u�u@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!n�m�m�@)ŏ1w-!_?1n�m�m�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��ZӼ�?!(�'�'�C@)_�Q�[?1�u�u�u
@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor/n��R?!���@)/n��R?1���@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!6�5�5��?)Ǻ���F?16�5�5��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI���+�X@Q�u��'j�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	F{��W3@F{��W3@!F{��W3@      ��!       "	��r�4�?��r�4�?!��r�4�?*      ��!       2	���Mb�?���Mb�?!���Mb�?:	��d�zh @��d�zh @!��d�zh @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q���+�X@y�u��'j�?