	�9�FP@�9�FP@!�9�FP@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�9�FP@H3MgQO@1uw��g�?A��+e�?IP6�
/@*	fffff&K@2U
Iterator::Model::ParallelMapV2���Q��?!Z]��ҟ;@)���Q��?1Z]��ҟ;@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?W[���?!�u:���;@)M�St$�?1E�~�p�4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeata��+e�?!��;_
�6@)M�O��?1����2@:Preprocessing2F
Iterator::Model��JY�8�?!�cTI�C@)S�!�uq{?1�D�~�(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!a��:�@)ŏ1w-!o?1a��:�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���{�?!���N@)_�Q�k?1"�O�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!qϸ�b�@)HP�s�b?1qϸ�b�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�J�4�?!�0&q�>@)_�Q�[?1"�O�	@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 96.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�3.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIc�N�C�X@Q��u�V��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	H3MgQO@H3MgQO@!H3MgQO@      ��!       "	uw��g�?uw��g�?!uw��g�?*      ��!       2	��+e�?��+e�?!��+e�?:	P6�
/@P6�
/@!P6�
/@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qc�N�C�X@y��u�V��?