	\Z�{�6@\Z�{�6@!\Z�{�6@	f!��o֖?f!��o֖?!f!��o֖?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6\Z�{�6@�ma4@1�)"�*�?A9��m4��?Ifj�!M@Y�Q���t?*	333333N@2U
Iterator::Model::ParallelMapV2�ݓ��Z�?!b*1��J?@)�ݓ��Z�?1b*1��J?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?W[���?!      9@)�{�Pk�?13n��[5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateŏ1w-!�?!���Tb*9@)������?1:�i�43@:Preprocessing2F
Iterator::Model �o_Ι?!�q˸e�D@) �o_�y?1�q˸e�$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip/n���?!i�4G�#M@)��H�}m?1?]��O�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��H�}m?!?]��O�@)��H�}m?1?]��O�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!i�4G�#@)/n��b?1i�4G�#@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapL7�A`�?!�E��`Q;@)��_�LU?1���o�7@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9f!��o֖?I5��0�X@Q2��g��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�ma4@�ma4@!�ma4@      ��!       "	�)"�*�?�)"�*�?!�)"�*�?*      ��!       2	9��m4��?9��m4��?!9��m4��?:	fj�!M@fj�!M@!fj�!M@B      ��!       J	�Q���t?�Q���t?!�Q���t?R      ��!       Z	�Q���t?�Q���t?!�Q���t?b      ��!       JGPUYf!��o֖?b q5��0�X@y2��g��?