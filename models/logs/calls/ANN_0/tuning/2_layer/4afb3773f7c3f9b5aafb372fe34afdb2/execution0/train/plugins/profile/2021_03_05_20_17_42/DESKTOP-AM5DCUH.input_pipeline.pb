	��j�	�3@��j�	�3@!��j�	�3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��j�	�3@��H��[2@1Pmp"���?A�3��7�?I�ӻx?��?*	������G@2U
Iterator::Model::ParallelMapV2��H�}�?!��@��@>@)��H�}�?1��@��@>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatF%u��?!�(��(�;@)�+e�X�?1���7@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap46<�R�?!T/�S/�6@)���Q�~?1:�:�/@:Preprocessing2F
Iterator::ModelˡE����?!�䚈E@)�HP�x?13X�3X�)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice_�Q�k?!$I�$I�@)_�Q�k?1$I�$I�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��6��?!e�ewL@)�~j�t�h?1�5�5@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!��@��@@)��H�}]?1��@��@@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��1e�X@Q��8��&�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��H��[2@��H��[2@!��H��[2@      ��!       "	Pmp"���?Pmp"���?!Pmp"���?*      ��!       2	�3��7�?�3��7�?!�3��7�?:	�ӻx?��?�ӻx?��?!�ӻx?��?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��1e�X@y��8��&�?