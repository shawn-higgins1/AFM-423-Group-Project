	Iط���2@Iط���2@!Iط���2@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-Iط���2@�J�ó�1@1@��"2��?Ag_y��"�?IX�2ı.�?*	433333L@2U
Iterator::Model::ParallelMapV2r�����?!���
b?@)r�����?1���
b?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?�ܵ�|�?!�D�)͋<@)�+e�X�?1��l�w64@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeata��+e�?!(Ni^�5@)/�$��?16�;��2@:Preprocessing2F
Iterator::Modela��+e�?!(Ni^�E@)�ZӼ�}?1��c-)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceU���N@s?!������ @)U���N@s?1������ @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipvq�-�?!�����L@)a��+ei?1(Ni^�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�?�pJ�
@)ŏ1w-!_?1�?�pJ�
@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 93.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�5.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�!b��X@Q��N4r�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�J�ó�1@�J�ó�1@!�J�ó�1@      ��!       "	@��"2��?@��"2��?!@��"2��?*      ��!       2	g_y��"�?g_y��"�?!g_y��"�?:	X�2ı.�?X�2ı.�?!X�2ı.�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�!b��X@y��N4r�?