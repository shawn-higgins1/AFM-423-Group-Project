	���XP2@���XP2@!���XP2@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���XP2@A�>�0@1t&m���?A��JY�8�?I����o�?*	43333sG@2U
Iterator::Model::ParallelMapV2���QI�?!x�0�}>@)���QI�?1x�0�}>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�(��0�?!7,s��9:@)�0�*�?1X�Q�,�5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap'�����?!Q�� ��6@)S�!�uq{?1$I�$I�,@:Preprocessing2F
Iterator::ModelQ�|a2�?!��wF@)-C��6z?1.;��J+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice����Mbp?!}/�ܼ!@)����Mbp?1}/�ܼ!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��<,Ԛ?!)��u��K@)-C��6j?1.;��J@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!}/�ܼ@)����Mb`?1}/�ܼ@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI����r�X@Q������?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	A�>�0@A�>�0@!A�>�0@      ��!       "	t&m���?t&m���?!t&m���?*      ��!       2	��JY�8�?��JY�8�?!��JY�8�?:	����o�?����o�?!����o�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q����r�X@y������?