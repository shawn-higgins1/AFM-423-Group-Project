	>���5@>���5@!>���5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails->���5@E�J!3@14iSu�l�?A�m�(�?I4�/.Ui @*	hffff�G@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�ݓ��Z�?!��d~�C@)�� �rh�?1#M�A@:Preprocessing2U
Iterator::Model::ParallelMapV2�]K�=�?!a��6�;@)�]K�=�?1a��6�;@:Preprocessing2F
Iterator::Model�l����?!=����YC@)��_�Lu?12N����%@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-C��6z?!P���:�*@)U���N@s?1.{CO�#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�X�� �?!�VAm�N@)Ǻ���f?1�gQ�Sn@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_�Q�[?!���s@)_�Q�[?1���s@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�0�*��?!��VAmE@)a2U0*�S?1|4!/l@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensora2U0*�S?!|4!/l@)a2U0*�S?1|4!/l@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!�gQ�Sn�?)Ǻ���F?1�gQ�Sn�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIl�tJ]�X@Q���b�h�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	E�J!3@E�J!3@!E�J!3@      ��!       "	4iSu�l�?4iSu�l�?!4iSu�l�?*      ��!       2	�m�(�?�m�(�?!�m�(�?:	4�/.Ui @4�/.Ui @!4�/.Ui @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb ql�tJ]�X@y���b�h�?