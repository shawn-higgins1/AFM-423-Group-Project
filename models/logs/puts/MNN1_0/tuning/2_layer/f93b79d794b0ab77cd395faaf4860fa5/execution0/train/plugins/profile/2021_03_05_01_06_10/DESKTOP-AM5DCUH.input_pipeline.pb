	ni5$��6@ni5$��6@!ni5$��6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-ni5$��6@B|`�i4@1DOʤ�6�?A
ףp=
�?I!�Ky�@*	������K@2U
Iterator::Model::ParallelMapV2�
F%u�?!ÂKe{�6@)�
F%u�?1ÂKe{�6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat2�%䃎?!���n�:@)��@��ǈ?1%D�9�5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateK�=�U�?!�<}���;@)Ǻ����?17���$4@:Preprocessing2F
Iterator::ModeljM�?!�1O-A@)-C��6z?1��/r�'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip㥛� ��?!�"gXpiP@)n��t?1p�l�:�!@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	�^)�p?!?ZMB@)	�^)�p?1?ZMB@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǺ���f?!7���$@)Ǻ���f?17���$@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�o_��?!�y�h5	>@)Ǻ���V?17���$@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��Q���X@QS��+��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	B|`�i4@B|`�i4@!B|`�i4@      ��!       "	DOʤ�6�?DOʤ�6�?!DOʤ�6�?*      ��!       2	
ףp=
�?
ףp=
�?!
ףp=
�?:	!�Ky�@!�Ky�@!!�Ky�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��Q���X@yS��+��?