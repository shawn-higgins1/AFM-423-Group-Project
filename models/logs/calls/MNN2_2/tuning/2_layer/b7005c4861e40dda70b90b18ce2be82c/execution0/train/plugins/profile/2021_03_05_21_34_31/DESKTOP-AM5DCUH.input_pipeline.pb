	��d�<�8@��d�<�8@!��d�<�8@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��d�<�8@@��w�6@1@�,��?Aı.n��?I�
��@*	�����K@2U
Iterator::Model::ParallelMapV2_�Qڋ?!�3���9@)_�Qڋ?1�3���9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatV-��?!(�h_��:@)�HP��?12�]�\�6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate��<,Ԋ?!f,1t+8@)HP�sׂ?1�W-0c�0@:Preprocessing2F
Iterator::ModelQ�|a2�?!�3��C@)�ZӼ�}?1m�}�3*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���x�&�?!*��ip�N@)"��u��q?1�Э8��@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice�q����o?!/R�D�@)�q����o?1/R�D�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!�W-0c�@)HP�s�b?1�W-0c�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap%u��?!��f,;@)-C��6Z?1��3���@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�^��͛X@Q!S����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	@��w�6@@��w�6@!@��w�6@      ��!       "	@�,��?@�,��?!@�,��?*      ��!       2	ı.n��?ı.n��?!ı.n��?:	�
��@�
��@!�
��@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�^��͛X@y!S����?