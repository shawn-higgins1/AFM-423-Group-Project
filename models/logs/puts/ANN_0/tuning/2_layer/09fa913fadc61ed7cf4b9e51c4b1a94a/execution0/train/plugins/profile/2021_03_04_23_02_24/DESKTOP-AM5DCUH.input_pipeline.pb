	�K⬄3@�K⬄3@!�K⬄3@	���GE�?���GE�?!���GE�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�K⬄3@+��<��1@1O#-��#�?A&�fe���?I�7�W�{�?Y���Je?*	�����M@2U
Iterator::Model::ParallelMapV2�o_��?!XeRY�<@)�o_��?1XeRY�<@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapL7�A`�?!�,�G9f<@)-C��6�?1�`�6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeaty�&1��?!�k��8@)������?1)�0�3@:Preprocessing2F
Iterator::Model=�U����?!Fmhw�D@)���_vO~?1�M�+y)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice���_vOn?!�M�+y@)���_vOn?1�M�+y@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�� �rh�?!񹒗�BM@)y�&1�l?1�k��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*�c?!+�
��@)a2U0*�c?1+�
��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���GE�?I��.6�X@QX�ܲ�;�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	+��<��1@+��<��1@!+��<��1@      ��!       "	O#-��#�?O#-��#�?!O#-��#�?*      ��!       2	&�fe���?&�fe���?!&�fe���?:	�7�W�{�?�7�W�{�?!�7�W�{�?B      ��!       J	���Je?���Je?!���Je?R      ��!       Z	���Je?���Je?!���Je?b      ��!       JGPUY���GE�?b q��.6�X@yX�ܲ�;�?