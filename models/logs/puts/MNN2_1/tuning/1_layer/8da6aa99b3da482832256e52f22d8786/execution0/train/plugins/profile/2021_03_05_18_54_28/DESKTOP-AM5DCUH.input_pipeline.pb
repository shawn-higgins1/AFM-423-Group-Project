	z����5@z����5@!z����5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-z����5@0�����3@1�Ҩ���?A�b�=y�?I: 	�v�?*	43333�J@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateM�O��?!i���B@)���&�?1և���XA@:Preprocessing2U
Iterator::Model::ParallelMapV2vq�-�?!Ȯ�dDP=@)vq�-�?1Ȯ�dDP=@:Preprocessing2F
Iterator::Model
ףp=
�?!��Vk:�D@)S�!�uq{?1�L��`�(@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�ZӼ�}?!97dWX*@)��_�Lu?1�R��K#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip2U0*��?!`��� M@)��_�Le?1�R��K@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!R��K3@)ŏ1w-!_?1R��K3@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��JY�8�?!��8+?!D@)�~j�t�X?14鏃qC@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor-C��6J?!|��h��?)-C��6J?1|��h��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!�S{��?)Ǻ���F?1�S{��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 90.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIɹ��X@QS�||��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	0�����3@0�����3@!0�����3@      ��!       "	�Ҩ���?�Ҩ���?!�Ҩ���?*      ��!       2	�b�=y�?�b�=y�?!�b�=y�?:	: 	�v�?: 	�v�?!: 	�v�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qɹ��X@yS�||��?