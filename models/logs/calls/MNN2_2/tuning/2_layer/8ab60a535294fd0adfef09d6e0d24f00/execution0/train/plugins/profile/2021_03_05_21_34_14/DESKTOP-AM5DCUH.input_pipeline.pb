	A�9w�Z7@A�9w�Z7@!A�9w�Z7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-A�9w�Z7@R~R���4@1�\4d<J�?A,e�X�?I�o%;v@*effff�H@)       =2U
Iterator::Model::ParallelMapV2���S㥋?!rY1P�;@)���S㥋?1rY1P�;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�
F%u�?!@R0���9@)�g��s��?1\����H5@:Preprocessing2F
Iterator::Model�g��s��?!\����HE@)� �	�?1���Bw�.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateA��ǘ��?!�C.+J6@)y�&1�|?1�����,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice	�^)�p?!ĉ��Bw @)	�^)�p?1ĉ��Bw @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���QI�?!�`2�L@)_�Q�k?1Yz'*O@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!�ˊ��@)�J�4a?1�ˊ��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�������?!tp�9@)Ǻ���V?1�dn}@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�����X@Q�Q�R��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	R~R���4@R~R���4@!R~R���4@      ��!       "	�\4d<J�?�\4d<J�?!�\4d<J�?*      ��!       2	,e�X�?,e�X�?!,e�X�?:	�o%;v@�o%;v@!�o%;v@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�����X@y�Q�R��?