	Ϣw*��5@Ϣw*��5@!Ϣw*��5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-Ϣw*��5@2Ƈ�˖3@1��n���?A�!H��?IY�|^� @*	533333M@2U
Iterator::Model::ParallelMapV2������?!~����G=@)������?1~����G=@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateŏ1w-!�?!p���:@)Zd;�O��?1#F��3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�?�߾�?!X�^�zu7@)Zd;�O��?1#F��3@:Preprocessing2F
Iterator::Model��@��ǘ?!��D@)�ZӼ�}?1*T�P(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vOn?!{��իW@)���_vOn?1{��իW@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip������?!~����GM@)a��+ei?1�s�Ν;@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�5�;Nё?!�lٲe�=@)/n��b?11bĈ#@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!1bĈ#@)/n��b?11bĈ#@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI4t�M4�X@Q����2�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	2Ƈ�˖3@2Ƈ�˖3@!2Ƈ�˖3@      ��!       "	��n���?��n���?!��n���?*      ��!       2	�!H��?�!H��?!�!H��?:	Y�|^� @Y�|^� @!Y�|^� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q4t�M4�X@y����2�?