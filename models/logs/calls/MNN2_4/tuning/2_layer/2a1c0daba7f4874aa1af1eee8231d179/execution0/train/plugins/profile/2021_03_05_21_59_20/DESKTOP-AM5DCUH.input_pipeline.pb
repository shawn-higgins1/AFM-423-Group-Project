	���҈e7@���҈e7@!���҈e7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���҈e7@>�٬��4@14�fI�?A����K�?I�4-��@*	������H@2U
Iterator::Model::ParallelMapV2�ZӼ��?!$I�$I�<@)�ZӼ��?1$I�$I�<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�������?!f�	��&9@)�0�*�?1R`��n�4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate������?!�X��W7@)� �	�?1��.�d�.@:Preprocessing2F
Iterator::Model�0�*�?!R`��n�D@)-C��6z?1�0�(�)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!I��ic@)�q����o?1I��ic@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip\ A�c̝?!��jA�FM@)F%u�k?1yv��1�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!P���˴@)/n��b?1P���˴@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�]K�=�?!V�H��:@)_�Q�[?1��C<;]@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIKu�($�X@Q+�b����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	>�٬��4@>�٬��4@!>�٬��4@      ��!       "	4�fI�?4�fI�?!4�fI�?*      ��!       2	����K�?����K�?!����K�?:	�4-��@�4-��@!�4-��@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qKu�($�X@y+�b����?