	ڐf�5@ڐf�5@!ڐf�5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-ڐf�5@z�I|3@1����,A�?A��ͪ�զ?I_EF$� @*	433333G@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�5�;Nё?!     �B@)����Mb�?1����=A@:Preprocessing2U
Iterator::Model::ParallelMapV2�
F%u�?!X�i��^;@)�
F%u�?1X�i��^;@:Preprocessing2F
Iterator::Model��~j�t�?!�{ayD@) �o_�y?1�{a�'+@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�HP�x?!����K*@)�J�4q?1���{"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�?�߾�?!�i�垆M@)�����g?1      @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!��{a@)ŏ1w-!_?1��{a@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapHP�sג?!�=����C@)����MbP?1����=@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor-C��6J?!a����?)-C��6J?1a����?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!��=���?)a2U0*�C?1��=���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIĭ�TٯX@Q�T��	�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	z�I|3@z�I|3@!z�I|3@      ��!       "	����,A�?����,A�?!����,A�?*      ��!       2	��ͪ�զ?��ͪ�զ?!��ͪ�զ?:	_EF$� @_EF$� @!_EF$� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qĭ�TٯX@y�T��	�?