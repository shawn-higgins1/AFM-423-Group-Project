	�i��&�7@�i��&�7@!�i��&�7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�i��&�7@���jH�4@1s��{�E�?A�HP��?I0��L��@*�����LG@)       =2U
Iterator::Model::ParallelMapV2��<,Ԋ?!���<@)��<,Ԋ?1���<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeattF��_�?!�J\�V�9@)��ׁsF�?1�yi�>5@:Preprocessing2F
Iterator::Model�0�*��?!>���E@)y�&1�|?1�W0��
.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate{�G�z�?!��k�u5@)F%u�{?1����S,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice_�Q�k?!�yi�>/@)_�Q�k?1�yi�>/@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�A`��"�?!�wO�nL@)�����g?1B$�=��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!�V���*@)����Mb`?1�V���*@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapg��j+��?!��x�w9@)_�Q�[?1�yi�>/@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��A��X@Q��oy�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���jH�4@���jH�4@!���jH�4@      ��!       "	s��{�E�?s��{�E�?!s��{�E�?*      ��!       2	�HP��?�HP��?!�HP��?:	0��L��@0��L��@!0��L��@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��A��X@y��oy�?