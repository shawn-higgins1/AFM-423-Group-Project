	2WՆ5@2WՆ5@!2WՆ5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-2WՆ5@����?K3@1�Ry=��?A�e��a��?I�g�K6��?*	33333�G@2U
Iterator::Model::ParallelMapV2-C��6�?!����;@)-C��6�?1����;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-C��6�?!����;@)��_vO�?1�=Q��6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateM�St$�?!Y�>��7@)���_vO~?1ȃ޺?9/@:Preprocessing2F
Iterator::ModelU���N@�?!��\��C@)�~j�t�x?1�ґ=Q)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!�Ȟ��t @)�q����o?1�Ȟ��t @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���QI�?! ��4+N@)-C��6j?1����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!l��Ӭ�@)����Mb`?1l��Ӭ�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap �o_Ή?!)�3�:@)��_�LU?1��Ӭ��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIF��D�X@Q�nL��n�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����?K3@����?K3@!����?K3@      ��!       "	�Ry=��?�Ry=��?!�Ry=��?*      ��!       2	�e��a��?�e��a��?!�e��a��?:	�g�K6��?�g�K6��?!�g�K6��?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qF��D�X@y�nL��n�?