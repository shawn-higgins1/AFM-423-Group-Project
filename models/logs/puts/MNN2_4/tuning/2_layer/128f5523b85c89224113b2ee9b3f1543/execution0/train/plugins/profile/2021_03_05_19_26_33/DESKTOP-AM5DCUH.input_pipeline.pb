	��)��R7@��)��R7@!��)��R7@	:M�r�_�?:M�r�_�?!:M�r�_�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��)��R7@�;O<g�4@1���Ft��?A�����?I��5>�� @Y[	�%qV�?*	�����H@2U
Iterator::Model::ParallelMapV2-C��6�?!ȩ�Xy�:@)-C��6�?1ȩ�Xy�:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�?�߾�?!��!�y{<@)g��j+��?11E>�S8@:Preprocessing2F
Iterator::Model�j+��ݓ?!�[��*D@)F%u�{?1��#]q+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenaten���?!����_4@)a��+ey?1z����)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��H�}m?!����@)��H�}m?1����@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipO��e�c�?!V��
l�M@)y�&1�l?1��6�$@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!
��ˡ@)����Mb`?1
��ˡ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapZd;�O��?!�����7@)_�Q�[?1d�o�@F@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9:M�r�_�?I�Ư�îX@Q^?��"��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�;O<g�4@�;O<g�4@!�;O<g�4@      ��!       "	���Ft��?���Ft��?!���Ft��?*      ��!       2	�����?�����?!�����?:	��5>�� @��5>�� @!��5>�� @B      ��!       J	[	�%qV�?[	�%qV�?![	�%qV�?R      ��!       Z	[	�%qV�?[	�%qV�?![	�%qV�?b      ��!       JGPUY:M�r�_�?b q�Ư�îX@y^?��"��?