	(CUL�g5@(CUL�g5@!(CUL�g5@	��,��?��,��?!��,��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6(CUL�g5@e�f�2@1�{�O��?A����K�?Iϣ���H@Y��h o��?*23333�H@)       =2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��d�`T�?!�Q��B@)���H�?1}���@@:Preprocessing2U
Iterator::Model::ParallelMapV2Ǻ����?!�j�o�6@)Ǻ����?1�j�o�6@:Preprocessing2F
Iterator::Model��d�`T�?!�Q��B@)S�!�uq{?1r~*! +@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateU���N@�?!t�KA3@)��0�*x?10>\���'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey�&1�l?!qE�W@)y�&1�l?1qE�W@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipT㥛� �?!�h}��O@)�����g?1��7V{@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!���1@)����Mb`?1���1@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap'�����?!��t�5@)��_�LU?1�P^Cy@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��,��?Im��
K�X@Q�Gi �`�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	e�f�2@e�f�2@!e�f�2@      ��!       "	�{�O��?�{�O��?!�{�O��?*      ��!       2	����K�?����K�?!����K�?:	ϣ���H@ϣ���H@!ϣ���H@B      ��!       J	��h o��?��h o��?!��h o��?R      ��!       Z	��h o��?��h o��?!��h o��?b      ��!       JGPUY��,��?b qm��
K�X@y�Gi �`�?