	�Q�U�M7@�Q�U�M7@!�Q�U�M7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�Q�U�M7@����h4@1���?�?Au����?Im�s�@*	�����LI@2U
Iterator::Model::ParallelMapV2�{�Pk�?!Wr{~9@)�{�Pk�?1Wr{~9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�<,Ԛ�?!v'���<@)�~j�t��?1��Pp%�7@:Preprocessing2F
Iterator::ModelˡE����?!n��O�AD@)ŏ1w-!?1�3[
.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�I+��?!����7�5@)y�&1�|?1R	�+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!�C��ܞ@)����Mbp?1�C��ܞ@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�:pΈҞ?!�0�:�M@)F%u�k?1!Y�B@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_�Le?!x�����@)��_�Le?1x�����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�HP��?!����T8@)a2U0*�S?1F�@���@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI? ��`�X@Q)���̧�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����h4@����h4@!����h4@      ��!       "	���?�?���?�?!���?�?*      ��!       2	u����?u����?!u����?:	m�s�@m�s�@!m�s�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q? ��`�X@y)���̧�?