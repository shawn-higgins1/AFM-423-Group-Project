	�>Ȳ`�5@�>Ȳ`�5@!�>Ȳ`�5@	r�(�{0�?r�(�{0�?!r�(�{0�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�>Ȳ`�5@\�M4O3@1�$��I�?A��ڊ�e�?I�܁:�@YW@��>�?*	     �F@2U
Iterator::Model::ParallelMapV2F%u��?!UUUUUU=@)F%u��?1UUUUUU=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�I+��?!�q�q8@);�O��n�?1      4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateM�St$�?!�q�q9@)�ZӼ�}?1�8��8�/@:Preprocessing2F
Iterator::ModeljM�?!��8��8E@)��0�*x?1��8��8*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�J�4q?!������"@)�J�4q?1������"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��ݓ���?!r�q�L@)��_vOf?1      @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!r�q�@)����Mb`?1r�q�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�HP��?!�q�q;@)��H�}M?1       @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9r�(�{0�?Ifm~ۡX@Q�b�F���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	\�M4O3@\�M4O3@!\�M4O3@      ��!       "	�$��I�?�$��I�?!�$��I�?*      ��!       2	��ڊ�e�?��ڊ�e�?!��ڊ�e�?:	�܁:�@�܁:�@!�܁:�@B      ��!       J	W@��>�?W@��>�?!W@��>�?R      ��!       Z	W@��>�?W@��>�?!W@��>�?b      ��!       JGPUYr�(�{0�?b qfm~ۡX@y�b�F���?