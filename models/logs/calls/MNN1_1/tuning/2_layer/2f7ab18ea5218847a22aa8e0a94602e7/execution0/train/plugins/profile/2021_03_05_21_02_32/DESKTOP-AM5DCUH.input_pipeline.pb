	�.�e�	7@�.�e�	7@!�.�e�	7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�.�e�	7@�*3���4@1bg
����?AEGr��?I:��KT� @*	    @H@2U
Iterator::Model::ParallelMapV22�%䃎?!�AG��>@)2�%䃎?1�AG��>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat46<�R�?!���fy6@)/n���?12���$2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�+e�X�?!/�~�Q�7@)�q����?1���0@:Preprocessing2F
Iterator::Model��_vO�?!�a�Y�DF@)S�!�uq{?1�_\��+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��H�}m?!�,O"Ӱ@)��H�}m?1�,O"Ӱ@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��q���?!Z�D�a�K@)a��+ei?1i�n�'�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!�/�~�Q@)�J�4a?1�/�~�Q@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�{�Pk�?!��fy�:@)�~j�t�X?1��AG�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��X@Q�^�{�G�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�*3���4@�*3���4@!�*3���4@      ��!       "	bg
����?bg
����?!bg
����?*      ��!       2	EGr��?EGr��?!EGr��?:	:��KT� @:��KT� @!:��KT� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��X@y�^�{�G�?