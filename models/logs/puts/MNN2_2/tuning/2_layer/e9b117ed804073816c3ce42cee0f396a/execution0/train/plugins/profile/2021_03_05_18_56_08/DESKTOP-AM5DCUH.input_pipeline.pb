	���
�6@���
�6@!���
�6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���
�6@S��:�3@1�ǚ�A�?A�+e�X�?I@3��x@*	ffffffF@2U
Iterator::Model::ParallelMapV2�+e�X�?!$I�$Ir9@)�+e�X�?1$I�$Ir9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�
F%u�?!�$I�$Y<@)��_vO�?1�m۶m8@:Preprocessing2F
Iterator::Model;�O��n�?!n۶m�D@)F%u�{?1n۶m�v-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateM�O��?!�m۶m�6@)�����w?1J�$I��)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice"��u��q?!%I�$I2#@)"��u��q?1%I�$I2#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipS�!�uq�?!�$I�$�M@)��_�Le?1n۶m�6@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!n۶m��@)ŏ1w-!_?1n۶m��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapZd;�O��?!�m۶m�9@)Ǻ���V?1      	@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 86.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��ѻ4�X@QR����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	S��:�3@S��:�3@!S��:�3@      ��!       "	�ǚ�A�?�ǚ�A�?!�ǚ�A�?*      ��!       2	�+e�X�?�+e�X�?!�+e�X�?:	@3��x@@3��x@!@3��x@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��ѻ4�X@yR����?