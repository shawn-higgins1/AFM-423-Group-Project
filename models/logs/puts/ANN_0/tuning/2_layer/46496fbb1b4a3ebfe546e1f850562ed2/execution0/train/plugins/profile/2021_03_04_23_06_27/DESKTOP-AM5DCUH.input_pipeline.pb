	s֧��4@s֧��4@!s֧��4@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-s֧��4@M�J�`2@1+�3��?A�z6�>�?I���f@*	������G@2U
Iterator::Model::ParallelMapV2vq�-�?!O��O��@@)vq�-�?1O��O��@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat������?!̧^̧^8@)a2U0*��?1t+t+4@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapǺ����?!������7@)�ZӼ�}?1��ԋ��-@:Preprocessing2F
Iterator::Modelw-!�l�?!\�\G@)�HP�x?13X�3X�)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice	�^)�p?!�:�:!@)	�^)�p?1�:�:!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipp_�Q�?!��~���J@)�����g?1̧^̧^@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!a��`��@)����Mb`?1a��`��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI^ E�X@Qc�7�nD�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	M�J�`2@M�J�`2@!M�J�`2@      ��!       "	+�3��?+�3��?!+�3��?*      ��!       2	�z6�>�?�z6�>�?!�z6�>�?:	���f@���f@!���f@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q^ E�X@yc�7�nD�?