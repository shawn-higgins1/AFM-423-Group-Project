	��{3@��{3@!��{3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��{3@`"��1@1XU/��d�?AtF��_�?I��!9���?*	����̌J@2U
Iterator::Model::ParallelMapV2�&S��?!�A�1#A@)�&S��?1�A�1#A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-C��6�?!Y}���8@)�g��s��?1��ʙ[�3@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaptF��_�?!�B�i6@)����Mb�?1���e�!.@:Preprocessing2F
Iterator::Modelc�ZB>�?!����G@)�ZӼ�}?1�w�*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�q����o?!���i�`@)�q����o?1���i�`@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�Pk�w�?!@�yQ-J@)a��+ei?1n�o�'Z@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!.&���@)/n��b?1.&���@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��c��X@Q8gR�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	`"��1@`"��1@!`"��1@      ��!       "	XU/��d�?XU/��d�?!XU/��d�?*      ��!       2	tF��_�?tF��_�?!tF��_�?:	��!9���?��!9���?!��!9���?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��c��X@y8gR�?