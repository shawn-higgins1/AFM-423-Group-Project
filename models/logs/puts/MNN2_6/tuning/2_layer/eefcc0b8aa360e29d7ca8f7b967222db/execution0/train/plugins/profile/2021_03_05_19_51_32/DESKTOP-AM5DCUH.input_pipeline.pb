	���[�7@���[�7@!���[�7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���[�7@Ĵo�~4@1�w��!�?AEGr��?I�	�i�l@*�����I@)       =2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatX9��v��?!dddddd>@)�]K�=�?1:@:Preprocessing2U
Iterator::Model::ParallelMapV2�g��s��?!������4@)�g��s��?1������4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateS�!�uq�?!GFFFFF:@)U���N@�?1onnnnn2@:Preprocessing2F
Iterator::ModelL7�A`�?!------@@)��0�*x?1######'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipD�l����?!jiiii�P@)"��u��q?1������ @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!______@)����Mbp?1______@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!AAAAAA@)/n��b?1AAAAAA@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap%u��?!������<@)��_�LU?1dddddd@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIH��9ǛX@Q�히1�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Ĵo�~4@Ĵo�~4@!Ĵo�~4@      ��!       "	�w��!�?�w��!�?!�w��!�?*      ��!       2	EGr��?EGr��?!EGr��?:	�	�i�l@�	�i�l@!�	�i�l@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qH��9ǛX@y�히1�?