	L�K�1"5@L�K�1"5@!L�K�1"5@	��B8\)�?��B8\)�?!��B8\)�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6L�K�1"5@�։��2@1A�! 8�?AZd;�O��?I�ZD� @Y���ʦ�?*	gffff�J@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate��ZӼ�?!�s��#C@)jM�?1�T���A@:Preprocessing2U
Iterator::Model::ParallelMapV2�o_��?!�\U?@)�o_��?1�\U?@:Preprocessing2F
Iterator::ModelZd;�O��?!1���s�E@) �o_�y?1�pIȣ'@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeata��+ey?!�b�C'@)	�^)�p?1O��N��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�sF���?!�Mf?�lL@)��_vOf?1�n�=C@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!���@)�J�4a?1���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap'�����?!'�F6D@)����MbP?1�쇑��?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensorǺ���F?!��\�?)Ǻ���F?1��\�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!y�����?)a2U0*�C?1y�����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��B8\)�?I'��ΤX@Qp�Y��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�։��2@�։��2@!�։��2@      ��!       "	A�! 8�?A�! 8�?!A�! 8�?*      ��!       2	Zd;�O��?Zd;�O��?!Zd;�O��?:	�ZD� @�ZD� @!�ZD� @B      ��!       J	���ʦ�?���ʦ�?!���ʦ�?R      ��!       Z	���ʦ�?���ʦ�?!���ʦ�?b      ��!       JGPUY��B8\)�?b q'��ΤX@yp�Y��?