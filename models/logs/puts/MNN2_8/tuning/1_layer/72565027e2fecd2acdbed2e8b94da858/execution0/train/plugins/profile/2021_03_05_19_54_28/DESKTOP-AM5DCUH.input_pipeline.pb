	�ZB>��7@�ZB>��7@!�ZB>��7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�ZB>��7@r�߅�U5@1��/�x��?Aa��+e�?IH�3�9@*	fffffFP@2U
Iterator::Model::ParallelMapV2�j+��ݓ?!�3���=@)�j+��ݓ?1�3���=@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateDio��ɔ?!/���.?@)	�^)ː?1�Ĥ�'19@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat2�%䃎?!,��K�6@)�������?1�\��33@:Preprocessing2F
Iterator::Model��ݓ���?!�Gy�C@)9��v��z?1���"�#@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!?���@)�q����o?1?���@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipn���?!n��N@)�����g?1dF����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*�c?!��u�:~@)a2U0*�c?1��u�:~@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap'�����?!���4�o@@)/n��R?1.$lg	�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�԰A�X@QY��'��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	r�߅�U5@r�߅�U5@!r�߅�U5@      ��!       "	��/�x��?��/�x��?!��/�x��?*      ��!       2	a��+e�?a��+e�?!a��+e�?:	H�3�9@H�3�9@!H�3�9@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�԰A�X@yY��'��?