	V�j-�5@V�j-�5@!V�j-�5@	TX$e�?TX$e�?!TX$e�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6V�j-�5@�� �m�3@1�Ǵ6���?AǺ���?I"�
�lC @Y��f���?*	      J@2U
Iterator::Model::ParallelMapV2"��u���?!�؉�؉@@)"��u���?1�؉�؉@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�(��0�?!vb'vb�7@)��_�L�?1      4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��0�*�?!;�;�6@)�J�4�?1vb'vb'0@:Preprocessing2F
Iterator::Model��+e�?!ى�؉]G@)�ZӼ�}?1��N��N+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice_�Q�k?!wb'vb'@)_�Q�k?1wb'vb'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��e�c]�?!'vb'v�J@){�G�zd?1�;�;@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�;�;@)ŏ1w-!_?1�;�;@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�{�Pk�?!��N���8@)/n��R?1�N��N� @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9TX$e�?I�Y1���X@Q��BP��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�� �m�3@�� �m�3@!�� �m�3@      ��!       "	�Ǵ6���?�Ǵ6���?!�Ǵ6���?*      ��!       2	Ǻ���?Ǻ���?!Ǻ���?:	"�
�lC @"�
�lC @!"�
�lC @B      ��!       J	��f���?��f���?!��f���?R      ��!       Z	��f���?��f���?!��f���?b      ��!       JGPUYTX$e�?b q�Y1���X@y��BP��?