	�F��7@�F��7@!�F��7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�F��7@=~o�G5@10�70�Q�?A�H�}�?IM�:�/+@*	      L@2U
Iterator::Model::ParallelMapV2Έ����?!�m۶m�@@)Έ����?1�m۶m�@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�
F%u�?!۶m۶�6@)'�����?1�m۶m3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��H�}�?!m۶m۶9@)��_�L�?1%I�$I�2@:Preprocessing2F
Iterator::Model�D���J�?!۶m۶F@)�HP�x?1�$I�$�%@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!$I�$I�@)����Mbp?1$I�$I�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipǺ���?!%I�$I�K@)�����g?1n۶m۶@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!$I�$I�@)����Mb`?1$I�$I�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap2U0*��?!�$I�$	<@)��_�LU?1%I�$I�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��0�̢X@Q���3�L�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	=~o�G5@=~o�G5@!=~o�G5@      ��!       "	0�70�Q�?0�70�Q�?!0�70�Q�?*      ��!       2	�H�}�?�H�}�?!�H�}�?:	M�:�/+@M�:�/+@!M�:�/+@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��0�̢X@y���3�L�?