	��g\8�7@��g\8�7@!��g\8�7@	��I�D��?��I�D��?!��I�D��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��g\8�7@�E�x�4@1�I}Yک�?A����K�?I�<dʇ�@Y�x�'er?*	�����YG@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate���{�?!�~�{�sA@)���Q��?1&sZY@@:Preprocessing2U
Iterator::Model::ParallelMapV2��<,Ԋ?!!�b(<@)��<,Ԋ?1!�b(<@:Preprocessing2F
Iterator::Model�j+��ݓ?!��AX�D@) �o_�y?1s�Z��*@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat_�Q�{?!�Xj
?-@)��ZӼ�t?1���Go�%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�#�����?!�7��:M@)a��+ei?1b�$/n�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_�Q�[?!�Xj
?@)_�Q�[?1�Xj
?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�5�;Nё?!���5*�B@)/n��R?1�S���@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensorǺ���F?!��S���?)Ǻ���F?1��S���?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!d�,چ��?)a2U0*�C?1d�,چ��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��I�D��?IX�iֿX@Q�l%�x�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�E�x�4@�E�x�4@!�E�x�4@      ��!       "	�I}Yک�?�I}Yک�?!�I}Yک�?*      ��!       2	����K�?����K�?!����K�?:	�<dʇ�@�<dʇ�@!�<dʇ�@B      ��!       J	�x�'er?�x�'er?!�x�'er?R      ��!       Z	�x�'er?�x�'er?!�x�'er?b      ��!       JGPUY��I�D��?b qX�iֿX@y�l%�x�?