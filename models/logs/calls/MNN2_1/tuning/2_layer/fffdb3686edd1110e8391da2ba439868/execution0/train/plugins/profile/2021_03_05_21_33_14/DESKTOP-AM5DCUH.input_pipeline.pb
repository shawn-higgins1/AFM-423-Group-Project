	n/�8@n/�8@!n/�8@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-n/�8@�)t^ck5@1�����?A�������?I�Q}�@*�����YH@)       =2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenatejM�?!�!@��C@)�5�;Nё?1j1SN�A@:Preprocessing2U
Iterator::Model::ParallelMapV2��<,Ԋ?!��[>�:@)��<,Ԋ?1��[>�:@:Preprocessing2F
Iterator::Modeln���?!��!@�D@)9��v��z?1X���*@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��0�*x?!��-��:(@)�q����o?1g��4 @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip\ A�c̝?!x[޿u�M@)�����g?1��Ľ�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!�\0�Vm@)����Mb`?1�\0�Vm@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapDio��ɔ?!�����D@)a2U0*�S?1l�h�@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor����MbP?!�\0�Vm @)����MbP?1�\0�Vm @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!SN�my��?)Ǻ���F?1SN�my��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI7_˺�X@Q\2(M �?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�)t^ck5@�)t^ck5@!�)t^ck5@      ��!       "	�����?�����?!�����?*      ��!       2	�������?�������?!�������?:	�Q}�@�Q}�@!�Q}�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q7_˺�X@y\2(M �?