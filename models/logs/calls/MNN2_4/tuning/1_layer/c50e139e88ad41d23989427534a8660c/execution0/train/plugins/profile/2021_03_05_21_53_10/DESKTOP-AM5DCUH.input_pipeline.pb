	�� ��D6@�� ��D6@!�� ��D6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�� ��D6@�ܙ	�{3@1J%<�ן�?A����z�?I���1G@*effff�H@)       =2U
Iterator::Model::ParallelMapV2%u��?!>�b��=@)%u��?1>�b��=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-C��6�?!%sp�9@)'�����?1B���f|5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenatetF��_�?!K/�D�7@)�q����?1W��FS/@:Preprocessing2F
Iterator::ModelˡE����?!6,���D@)�����w?1[�]K'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	�^)�p?!ĉ��Bw @)	�^)�p?1ĉ��Bw @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�X�� �?!��Bw�jM@)�~j�t�h?1�k��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!�ˊ��@)�J�4a?1�ˊ��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���S㥋?!rY1P�;@)-C��6Z?1%sp�	@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�Z6�8�X@Q@��d���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�ܙ	�{3@�ܙ	�{3@!�ܙ	�{3@      ��!       "	J%<�ן�?J%<�ן�?!J%<�ן�?*      ��!       2	����z�?����z�?!����z�?:	���1G@���1G@!���1G@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�Z6�8�X@y@��d���?