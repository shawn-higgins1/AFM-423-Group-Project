	���5@���5@!���5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���5@���컎2@1�� �K�?A��&��?I�Ye���@*	�����H@2U
Iterator::Model::ParallelMapV2S�!�uq�?!0�/�;@)S�!�uq�?10�/�;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�������?!��9@)��_�L�?1Ʀ�Ŧ�5@:Preprocessing2F
Iterator::ModelDio��ɔ?!��E@)lxz�,C|?1�u��u�,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate/�$��?!t��s��5@) �o_�y?1�$�$*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�J�4q?!)im)im!@)�J�4q?1)im)im!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip6�;Nё�?!� �� �L@)��H�}m?1�!��!�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!)im)im@)�J�4a?1)im)im@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��0�*�?!L8{L8{8@)��_�LU?1Ʀ�Ŧ�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��l�X@Q���}�I�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���컎2@���컎2@!���컎2@      ��!       "	�� �K�?�� �K�?!�� �K�?*      ��!       2	��&��?��&��?!��&��?:	�Ye���@�Ye���@!�Ye���@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��l�X@y���}�I�?