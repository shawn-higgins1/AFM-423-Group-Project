	�`�.�4@�`�.�4@!�`�.�4@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�`�.�4@�\5�i2@1!�˛��?A46<�R�?I�hr1V�?*	�����YP@2U
Iterator::Model::ParallelMapV2Dio��ɔ?!�Ph-
?@)Dio��ɔ?1�Ph-
?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��&��?!�����A@)jM�?1G���s4=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeattF��_�?!"$�22@)��ׁsF�?1h�bnuF.@:Preprocessing2F
Iterator::Model��q���?! �ԐD@)F%u�{?1��A��.$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice����Mbp?!p��7�v@)����Mbp?1p��7�v@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��ʡE��?!���+oM@)y�&1�l?1�֢Ph@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!p��7�v@)����Mb`?1p��7�v@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 91.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�7.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI箢��X@Q�:FTW�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�\5�i2@�\5�i2@!�\5�i2@      ��!       "	!�˛��?!�˛��?!!�˛��?*      ��!       2	46<�R�?46<�R�?!46<�R�?:	�hr1V�?�hr1V�?!�hr1V�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q箢��X@y�:FTW�?