	��$�J7@��$�J7@!��$�J7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��$�J7@{�f�l�4@1Ę��Rx�?A�e��a��?I�o`r��@*	������I@2U
Iterator::Model::ParallelMapV2S�!�uq�?!���3�9@)S�!�uq�?1���3�9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateL7�A`�?!�l�w6�?@) �o_Ή?1��2R_8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatǺ����?!n:�eש5@)HP�sׂ?1��7��1@:Preprocessing2F
Iterator::Model��ׁsF�?!]AL� &C@)-C��6z?15'��(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!Ь���,@)�q����o?1Ь���,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�|a2U�?!���Q��N@)�~j�t�h?1Q�T�5@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!��pl��@)����Mb`?1��pl��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapr�����?!��|4A@)a2U0*�S?1�V���@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��j�X@Q�R���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	{�f�l�4@{�f�l�4@!{�f�l�4@      ��!       "	Ę��Rx�?Ę��Rx�?!Ę��Rx�?*      ��!       2	�e��a��?�e��a��?!�e��a��?:	�o`r��@�o`r��@!�o`r��@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��j�X@y�R���?