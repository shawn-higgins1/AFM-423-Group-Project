	e��2��5@e��2��5@!e��2��5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-e��2��5@��B63@1�����?AV����_�?I�sD�KI@*�����?H@)       =2U
Iterator::Model::ParallelMapV2�Pk�w�?!�W?�<@)�Pk�w�?1�W?�<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat �o_Ή?!CG���9@)/�$��?1b�Y�D�5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateM�St$�?!�v�<�L7@)���Q�~?1gy���.@:Preprocessing2F
Iterator::ModelM�O��?!��,O"�D@) �o_�y?1CG���)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceŏ1w-!o?!?��W@)ŏ1w-!o?1?��W@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipB>�٬��?!P"Ӱ�,M@)a��+ei?1i�n�'�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!�/�~�Q@)�J�4a?1�/�~�Q@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap �o_Ή?!CG���9@)��_�LU?1���
|q@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIeTѦ�X@Q���~�,�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��B63@��B63@!��B63@      ��!       "	�����?�����?!�����?*      ��!       2	V����_�?V����_�?!V����_�?:	�sD�KI@�sD�KI@!�sD�KI@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qeTѦ�X@y���~�,�?