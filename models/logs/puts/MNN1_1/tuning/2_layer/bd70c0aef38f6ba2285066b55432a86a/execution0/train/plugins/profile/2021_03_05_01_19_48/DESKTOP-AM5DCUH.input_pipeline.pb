	���֑7@���֑7@!���֑7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���֑7@|�_���4@1����?A,e�X�?IS���@*	������J@2U
Iterator::Model::ParallelMapV2vq�-�?!�¯�Dz=@)vq�-�?1�¯�Dz=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�HP��?!�����6@)�0�*�?1L�*g73@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate9��v���?!�_���@8@)/n���?1���-�j0@:Preprocessing2F
Iterator::Model���Mb�?!*g���E@)� �	�?1c"=P9�,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�J�4q?!T�n�W@)�J�4q?1T�n�W@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�:pΈҞ?!�՘HL@)���_vOn?1�1���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!&���[@)ŏ1w-!_?1&���[@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�ZӼ��?!�@��~:@)a2U0*�S?1v��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI.��;.�X@Q��q��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	|�_���4@|�_���4@!|�_���4@      ��!       "	����?����?!����?*      ��!       2	,e�X�?,e�X�?!,e�X�?:	S���@S���@!S���@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q.��;.�X@y��q��?