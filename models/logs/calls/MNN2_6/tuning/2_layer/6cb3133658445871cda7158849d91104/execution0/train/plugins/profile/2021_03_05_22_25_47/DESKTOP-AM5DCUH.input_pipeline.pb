	����G�3@����G�3@!����G�3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-����G�3@�E� "2@1$C��g�?A�(��0�?I�u�|�H�?*fffff�I@)       =2U
Iterator::Model::ParallelMapV2�ZӼ��?!�vط�i;@)�ZӼ��?1�vط�i;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatV-��?!�_5��;@)�(��0�?1�[�þ7@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��0�*�?!z,���6@)ŏ1w-!?1�ո�W-@:Preprocessing2F
Iterator::ModelM�O��?!�W�{�C@)�~j�t�x?1�r�~�*'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipvq�-�?!�r�~�N@)�g��s�u?1j�}+�v$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�J�4q?!v�)�Y7 @)�J�4q?1v�)�Y7 @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!�P��@)/n��b?1�P��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�D�Q�X@Q��.x���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�E� "2@�E� "2@!�E� "2@      ��!       "	$C��g�?$C��g�?!$C��g�?*      ��!       2	�(��0�?�(��0�?!�(��0�?:	�u�|�H�?�u�|�H�?!�u�|�H�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�D�Q�X@y��.x���?