	<���$7@<���$7@!<���$7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-<���$7@�f�W�4@1�P�n��?A,e�X�?I��ډ�� @*	43333�H@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate46<��?!�{u90VB@)�J�4�?1�%�8k�@@:Preprocessing2U
Iterator::Model::ParallelMapV2���S㥋?!(<	�;@)���S㥋?1(<	�;@:Preprocessing2F
Iterator::Model�N@aÓ?!l_"��VC@)�����w?1aq*?'@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatŏ1w-!?!9�T��u.@)�+e�Xw?1�|�:�&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipK�=�U�?!���pL�N@)y�&1�l?1���^]@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!9�T��u@)ŏ1w-!_?19�T��u@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap+�����?!�Y��C@)a2U0*�S?1?�]�=@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor-C��6J?!�-}Ļ��?)-C��6J?1�-}Ļ��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!��K�q�?)Ǻ���F?1��K�q�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI)#�}�X@Q�57����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�f�W�4@�f�W�4@!�f�W�4@      ��!       "	�P�n��?�P�n��?!�P�n��?*      ��!       2	,e�X�?,e�X�?!,e�X�?:	��ډ�� @��ډ�� @!��ډ�� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q)#�}�X@y�57����?