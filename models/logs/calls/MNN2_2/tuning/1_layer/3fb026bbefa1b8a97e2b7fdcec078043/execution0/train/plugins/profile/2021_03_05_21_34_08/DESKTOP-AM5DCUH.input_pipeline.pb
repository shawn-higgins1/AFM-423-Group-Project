	������5@������5@!������5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-������5@��3ڪ�3@1�0���?A�H.�!��?I	l��3A@*	gffff&F@2U
Iterator::Model::ParallelMapV2�(��0�?!!�ǒ��;@)�(��0�?1!�ǒ��;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatǺ����?!8m<H9@)��~j�t�?1�l��q5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate46<�R�?! K�ߚ8@)_�Q�{?1~�h$�.@:Preprocessing2F
Iterator::Model;�O��n�?!�÷&�PD@)�+e�Xw?1W+Ouϻ)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice	�^)�p?!��/���"@)	�^)�p?1��/���"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip	�c�?!<H��M@)Ǻ���f?18m<H@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_�Q�[?!~�h$�@)_�Q�[?1~�h$�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�(��0�?!!�ǒ��;@)Ǻ���V?18m<H	@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�A@`t�X@Q�;���E�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��3ڪ�3@��3ڪ�3@!��3ڪ�3@      ��!       "	�0���?�0���?!�0���?*      ��!       2	�H.�!��?�H.�!��?!�H.�!��?:		l��3A@	l��3A@!	l��3A@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�A@`t�X@y�;���E�?