	���k��7@���k��7@!���k��7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���k��7@$d �.�4@1]���l��?A|�Pk��?I�#���K@*	�����K@2U
Iterator::Model::ParallelMapV2_�Qڋ?!�3���9@)_�Qڋ?1�3���9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatX9��v��?!F�x��<@)��<,Ԋ?1f,1t+8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�]K�=�?!9��7�8@)��ׁsF�?1;/R�D2@:Preprocessing2F
Iterator::Model���<,�?!G��f,B@)�HP�x?12�]�\�&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipD�l����?!�U���O@)"��u��q?1�Э8��@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice_�Q�k?!�3���@)_�Q�k?1�3���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*�c?!�f=Q�@)a2U0*�c?1�f=Q�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap%u��?!��f,;@)Ǻ���V?1vMr	�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�k�4��X@Q�%�2�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	$d �.�4@$d �.�4@!$d �.�4@      ��!       "	]���l��?]���l��?!]���l��?*      ��!       2	|�Pk��?|�Pk��?!|�Pk��?:	�#���K@�#���K@!�#���K@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�k�4��X@y�%�2�?