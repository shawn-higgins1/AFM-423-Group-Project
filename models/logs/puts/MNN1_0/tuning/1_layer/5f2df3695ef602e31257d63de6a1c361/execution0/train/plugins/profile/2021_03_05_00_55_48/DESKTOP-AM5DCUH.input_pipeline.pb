	���̓�7@���̓�7@!���̓�7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���̓�7@.8��_ 5@1ݘ����?A��0�*�?I�g�o}�@*	������P@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�j+��ݓ?!������<@)�J�4�?1������8@:Preprocessing2U
Iterator::Model::ParallelMapV2%u��?!�5@)%u��?1�5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate"��u���?!KKKKKK9@)tF��_�?1     �1@:Preprocessing2F
Iterator::ModeltF��_�?!     �A@)�&S��?1������*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�e��a��?!     @P@)�g��s�u?1------@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�g��s�u?!------@)�g��s�u?1------@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_�Le?!������@)��_�Le?1������@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap+�����?!-----�<@)HP�s�b?1@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI����d�X@Q�X��&�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	.8��_ 5@.8��_ 5@!.8��_ 5@      ��!       "	ݘ����?ݘ����?!ݘ����?*      ��!       2	��0�*�?��0�*�?!��0�*�?:	�g�o}�@�g�o}�@!�g�o}�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q����d�X@y�X��&�?