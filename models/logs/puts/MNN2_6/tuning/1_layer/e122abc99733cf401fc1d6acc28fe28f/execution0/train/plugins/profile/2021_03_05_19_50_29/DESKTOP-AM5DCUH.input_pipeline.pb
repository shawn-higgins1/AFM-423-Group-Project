	�(�A&�5@�(�A&�5@!�(�A&�5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�(�A&�5@/��|� 3@1���uo�?A��ͪ�զ?I��,�X@*	33333�G@2U
Iterator::Model::ParallelMapV2y�&1��?!�a�3A:=@)y�&1��?1�a�3A:=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�(��0�?!������9@)��_�L�?1�^�?�5@:Preprocessing2F
Iterator::ModelQ�|a2�?!8Ii%��E@)S�!�uq{?1Da�-��+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate46<�R�?!a4"@x�6@)9��v��z?1?����#+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice/n��r?!����!_"@)/n��r?1����!_"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip_�Qڛ?!ȶ��ydL@)��_vOf?1�	�i�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�b�?��@)ŏ1w-!_?1�b�?��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�HP��?!4`4"@x9@)��_�LU?1�^�?�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIQ)�m�X@Q����$�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	/��|� 3@/��|� 3@!/��|� 3@      ��!       "	���uo�?���uo�?!���uo�?*      ��!       2	��ͪ�զ?��ͪ�զ?!��ͪ�զ?:	��,�X@��,�X@!��,�X@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qQ)�m�X@y����$�?