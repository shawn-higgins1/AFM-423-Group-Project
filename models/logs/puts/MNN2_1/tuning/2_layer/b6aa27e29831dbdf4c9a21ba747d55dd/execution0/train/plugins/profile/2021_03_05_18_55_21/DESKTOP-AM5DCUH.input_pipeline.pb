	���Y.S7@���Y.S7@!���Y.S7@	��ۉ�ߝ?��ۉ�ߝ?!��ۉ�ߝ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6���Y.S7@ݶ�Q�4@10�AC��?A(~��k	�?I1A��@Y:@0G��{?*	      G@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate"��u���?!Y�B�B@)2U0*��?1Y�B�A@:Preprocessing2U
Iterator::Model::ParallelMapV2M�St$�?!Y�B��8@)M�St$�?1Y�B��8@:Preprocessing2F
Iterator::Model�5�;Nё?!���7��B@)�HP�x?1�B���*@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeaty�&1�|?!�7��Mo.@){�G�zt?1����7�%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���QI�?!!Y�BO@)a��+ei?1ozӛ��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!���,d@)����Mb`?1���,d@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�l����?!�Moz�D@)��_�LU?1�Mozӛ@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��H�}M?!��7��M�?)��H�}M?1��7��M�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!Nozӛ��?)a2U0*�C?1Nozӛ��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��ۉ�ߝ?Ibl��s�X@Q�x=^�+�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	ݶ�Q�4@ݶ�Q�4@!ݶ�Q�4@      ��!       "	0�AC��?0�AC��?!0�AC��?*      ��!       2	(~��k	�?(~��k	�?!(~��k	�?:	1A��@1A��@!1A��@B      ��!       J	:@0G��{?:@0G��{?!:@0G��{?R      ��!       Z	:@0G��{?:@0G��{?!:@0G��{?b      ��!       JGPUY��ۉ�ߝ?b qbl��s�X@y�x=^�+�?