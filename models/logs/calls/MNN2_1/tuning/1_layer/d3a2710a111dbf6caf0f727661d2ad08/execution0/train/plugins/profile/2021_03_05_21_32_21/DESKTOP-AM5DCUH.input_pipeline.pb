	�Վ�%6@�Վ�%6@!�Վ�%6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�Վ�%6@�<|3@1m���?A�ׁsF��?I>�ɋL@*	������G@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateΈ����?!A��@��C@)�J�4�?1������A@:Preprocessing2U
Iterator::Model::ParallelMapV2�
F%u�?!�@��@�:@)�
F%u�?1�@��@�:@:Preprocessing2F
Iterator::Modela2U0*��?!t+t+D@)9��v��z?1��O��O+@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�HP�x?!3X�3X�)@)"��u��q?1�5�5"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�ZӼ��?!��ԋ��M@)��_�Le?1�F��F�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!��@��@@)��H�}]?1��@��@@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���<,�?!�z1�z�D@)/n��R?1��|��|@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor/n��R?!��|��|@)/n��R?1��|��|@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!�������?)Ǻ���F?1�������?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI����E�X@Q�]�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�<|3@�<|3@!�<|3@      ��!       "	m���?m���?!m���?*      ��!       2	�ׁsF��?�ׁsF��?!�ׁsF��?:	>�ɋL@>�ɋL@!>�ɋL@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q����E�X@y�]�?