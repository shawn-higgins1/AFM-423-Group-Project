	�lu9%�8@�lu9%�8@!�lu9%�8@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�lu9%�8@B��ފL6@1�|?5^��?A&S���?I'��d�@*	������F@2U
Iterator::Model::ParallelMapV2-C��6�?!�Ź�Q<@)-C��6�?1�Ź�Q<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�+e�X�?!2�s�89@)U���N@�?1M�l���4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate/�$��?!q�c�:7@)lxz�,C|?1�~H��.@:Preprocessing2F
Iterator::Model�ݓ��Z�?!f�&_6�D@)�HP�x?1��	��*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��H�}m?!	�����@)��H�}m?1	�����@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip	�c�?!�|٠�M@)-C��6j?1�Ź�Q@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!�/4��@)����Mb`?1�/4��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapg��j+��?!�(�u��9@)a2U0*�S?1��S+=@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI+Pd�/�X@QZ��� ��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	B��ފL6@B��ފL6@!B��ފL6@      ��!       "	�|?5^��?�|?5^��?!�|?5^��?*      ��!       2	&S���?&S���?!&S���?:	'��d�@'��d�@!'��d�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q+Pd�/�X@yZ��� ��?