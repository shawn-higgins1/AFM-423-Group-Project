	�� u7@�� u7@!�� u7@	QH���?QH���?!QH���?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�� u7@B?S�[�4@1�3���l�?A=�U����?I.Y�&�@YV��L�p?*	������I@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenatej�t��?!�N��N�D@)8��d�`�?1'vb'v"C@:Preprocessing2U
Iterator::Model::ParallelMapV2������?!��N��N6@)������?1��N��N6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�J�4�?!vb'vb'0@)�~j�t�x?1;�;�'@:Preprocessing2F
Iterator::Model�5�;Nё?!�;��@@)�����w?1��N��N&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipe�X��?!'vb'v�P@)_�Q�k?1wb'vb'@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*�c?!c'vb'v@)a2U0*�c?1c'vb'v@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�+e�X�?!�N��N�E@)��_�LU?1      @:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��H�}M?!;�;��?)��H�}M?1;�;��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!�؉�؉�?)Ǻ���F?1�؉�؉�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9QH���?I�Ū�X@Q�^�����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	B?S�[�4@B?S�[�4@!B?S�[�4@      ��!       "	�3���l�?�3���l�?!�3���l�?*      ��!       2	=�U����?=�U����?!=�U����?:	.Y�&�@.Y�&�@!.Y�&�@B      ��!       J	V��L�p?V��L�p?!V��L�p?R      ��!       Z	V��L�p?V��L�p?!V��L�p?b      ��!       JGPUYQH���?b q�Ū�X@y�^�����?