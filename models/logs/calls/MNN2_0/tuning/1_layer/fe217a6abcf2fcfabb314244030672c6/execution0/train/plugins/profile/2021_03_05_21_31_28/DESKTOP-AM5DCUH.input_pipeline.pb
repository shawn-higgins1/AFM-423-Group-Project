	����w�5@����w�5@!����w�5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-����w�5@��.ޏ+3@1��M��?A����z�?I>����� @*effff�H@)       =2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate���<,�?!��<-t�C@)��y�):�?1p��A@:Preprocessing2U
Iterator::Model::ParallelMapV2�Pk�w�?!ݱ�a�;@)�Pk�w�?1ݱ�a�;@:Preprocessing2F
Iterator::ModelM�O��?!��l��GD@) �o_�y?1Y1P�M)@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-C��6z?!%sp�)@)HP�s�r?1�ҋ8Qy"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���_vO�?!#�9�M@){�G�zd?1�ٌ�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!���h�@)��H�}]?1���h�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapQ�|a2�?!M�_{�D@)����MbP?1�G
&s @:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor����MbP?!�G
&s @)����MbP?1�G
&s @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice��H�}M?!���h��?)��H�}M?1���h��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIS9�X@Q�p�vqc�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��.ޏ+3@��.ޏ+3@!��.ޏ+3@      ��!       "	��M��?��M��?!��M��?*      ��!       2	����z�?����z�?!����z�?:	>����� @>����� @!>����� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qS9�X@y�p�vqc�?