	]p��6@]p��6@!]p��6@	F��9	�?F��9	�?!F��9	�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6]p��6@K�^b\4@1�Z^��6�?A�E���Ԩ?I�bFx{p@Y�bFx{p?*	      G@2U
Iterator::Model::ParallelMapV2�������?!B���,;@)�������?1B���,;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat �o_Ή?!���,d;@)��_vO�?18��Moz7@:Preprocessing2F
Iterator::Model�N@aÓ?!8��Mo�D@)_�Q�{?1Y�B��-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateM�O��?!pzӛ��5@)�~j�t�x?1!Y�B*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	�^)�p?!��Moz�!@)	�^)�p?1��Moz�!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipz6�>W�?!�B��M@)�����g?1ӛ���7@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!��7��M@)��H�}]?1��7��M@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapǺ����?!�,d!Y8@)/n��R?1���,d!@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9F��9	�?I-I@{K�X@Q�pa��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	K�^b\4@K�^b\4@!K�^b\4@      ��!       "	�Z^��6�?�Z^��6�?!�Z^��6�?*      ��!       2	�E���Ԩ?�E���Ԩ?!�E���Ԩ?:	�bFx{p@�bFx{p@!�bFx{p@B      ��!       J	�bFx{p?�bFx{p?!�bFx{p?R      ��!       Z	�bFx{p?�bFx{p?!�bFx{p?b      ��!       JGPUYF��9	�?b q-I@{K�X@y�pa��?