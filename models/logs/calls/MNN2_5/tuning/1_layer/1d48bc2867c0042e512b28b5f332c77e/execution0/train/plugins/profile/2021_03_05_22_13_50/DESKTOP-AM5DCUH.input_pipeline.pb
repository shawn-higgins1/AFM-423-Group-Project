	(__�2@(__�2@!(__�2@	?~��??~��?!?~��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6(__�2@�6qr�O1@1��6�4D�?AI.�!���?I��66;R�?Y���e��?*	43333sH@2U
Iterator::Model::ParallelMapV2V-��?!5<� �=@)V-��?15<� �=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-C��6�?!A�9�-:@)'�����?1�#\Т�5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapg��j+��?!W��C'�7@)vq�-�?1��ߍ�'0@:Preprocessing2F
Iterator::Model䃞ͪϕ?!��u�E@)_�Q�{?1�/]��+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceŏ1w-!o?!lb��v@)ŏ1w-!o?1lb��v@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Ziplxz�,C�?!�f5�8L@)a��+ei?1��'��[@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!��uǋ-@)�J�4a?1��uǋ-@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?~��?I�QIi��X@Q��n�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�6qr�O1@�6qr�O1@!�6qr�O1@      ��!       "	��6�4D�?��6�4D�?!��6�4D�?*      ��!       2	I.�!���?I.�!���?!I.�!���?:	��66;R�?��66;R�?!��66;R�?B      ��!       J	���e��?���e��?!���e��?R      ��!       Z	���e��?���e��?!���e��?b      ��!       JGPUY?~��?b q�QIi��X@y��n�?