	$`tys�H@$`tys�H@!$`tys�H@	��!�d��?��!�d��?!��!�d��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6$`tys�H@/��$#G@1�\T��?AM�St$�?I�pvk�,@Y�uT5A�?*	����̌G@2U
Iterator::Model::ParallelMapV2y�&1��?!u��W�=@)y�&1��?1u��W�=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatZd;�O��?!���j8@)a2U0*��?1����a4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate46<�R�?!	��j$7@)�<,Ԛ�}?1x��t�.@:Preprocessing2F
Iterator::Model��ZӼ�?!C���E@)-C��6z?1"��-+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��H�}m?!�O[h��@)��H�}m?1�O[h��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipz6�>W�?!��iXL@)�~j�t�h?1@m�Kz@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!d���"@)ŏ1w-!_?1d���"@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�HP��?!x��t��9@)��_�LU?1|+�g�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 93.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�5.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��!�d��?IW����X@Q�h�4��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	/��$#G@/��$#G@!/��$#G@      ��!       "	�\T��?�\T��?!�\T��?*      ��!       2	M�St$�?M�St$�?!M�St$�?:	�pvk�,@�pvk�,@!�pvk�,@B      ��!       J	�uT5A�?�uT5A�?!�uT5A�?R      ��!       Z	�uT5A�?�uT5A�?!�uT5A�?b      ��!       JGPUY��!�d��?b qW����X@y�h�4��?