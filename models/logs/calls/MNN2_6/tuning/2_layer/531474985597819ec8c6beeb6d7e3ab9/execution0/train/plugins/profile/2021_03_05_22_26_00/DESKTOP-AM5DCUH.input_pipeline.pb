	u���3@u���3@!u���3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-u���3@V�a�B2@1����q��?A0*��D�?I5
If��?*	�����YM@2U
Iterator::Model::ParallelMapV2Dio��ɔ?!�O;h�JA@)Dio��ɔ?1�O;h�JA@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��H�}�?!%��q�8@)��@��ǈ?1�I�o �4@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapV-��?!�ߝ��8@)�I+��?1GZ�6F�2@:Preprocessing2F
Iterator::Model���S㥛?!�K�Z��F@)S�!�uq{?1M�����&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicey�&1�l?!��.��@)y�&1�l?1��.��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���~�:�?!�I�o K@)Ǻ���f?1r�X@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!��q�X@)HP�s�b?1��q�X@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 91.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�7.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�<U�X@Q\@��*��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	V�a�B2@V�a�B2@!V�a�B2@      ��!       "	����q��?����q��?!����q��?*      ��!       2	0*��D�?0*��D�?!0*��D�?:	5
If��?5
If��?!5
If��?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�<U�X@y\@��*��?