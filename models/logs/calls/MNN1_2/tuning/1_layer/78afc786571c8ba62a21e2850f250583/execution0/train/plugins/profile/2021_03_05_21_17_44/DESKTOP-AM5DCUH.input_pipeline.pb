	�'֩�}5@�'֩�}5@!�'֩�}5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�'֩�}5@�a���(3@1	oB@��?A&S���?IN�w(
� @*�����YH@)       =2U
Iterator::Model::ParallelMapV2� �	��?!�K�F�?@)� �	��?1�K�F�?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat������?!��Ľ�7@)��~j�t�?1�my�ց3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateǺ����?!SN�my�6@)� �	�?1�K�F�/@:Preprocessing2F
Iterator::Model��_vO�?!��5-F@)a��+ey?1 )~pFv)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey�&1�l?!�T�W�@)y�&1�l?1�T�W�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��6��?!�OX���K@)�����g?1��Ľ�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!��e�?@)�J�4a?1��e�?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap �o_Ή?!���h�9@)Ǻ���V?1SN�my�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��4���X@Q&����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�a���(3@�a���(3@!�a���(3@      ��!       "		oB@��?	oB@��?!	oB@��?*      ��!       2	&S���?&S���?!&S���?:	N�w(
� @N�w(
� @!N�w(
� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��4���X@y&����?