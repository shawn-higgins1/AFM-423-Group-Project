	8�0C�7@8�0C�7@!8�0C�7@	]G'd�?]G'd�?!]G'd�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails68�0C�7@ �={�4@1 �)U���?A�!H��?I��X���@YN)���]r?*	23333sH@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate���&�?!��4�C@)�J�4�?1��uǋ-A@:Preprocessing2U
Iterator::Model::ParallelMapV2�{�Pk�?!l	��_a:@)�{�Pk�?1l	��_a:@:Preprocessing2F
Iterator::Model���<,�?!�Pvn�$D@)_�Q�{?1�/]��+@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-C��6z?!A�9�-*@);�O��nr?1�}��g"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�<,Ԛ�?!`���Z�M@)Ǻ���f?1xc�	e�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!lb��v@)ŏ1w-!_?1lb��v@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap{�G�z�?!b��,sD@)��_�LU?1�έ�D@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice����MbP?!��c�#\ @)����MbP?1��c�#\ @:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��H�}M?!ɀz�r�?)��H�}M?1ɀz�r�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9]G'd�?I���`�X@QB;�
=��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	 �={�4@ �={�4@! �={�4@      ��!       "	 �)U���? �)U���?! �)U���?*      ��!       2	�!H��?�!H��?!�!H��?:	��X���@��X���@!��X���@B      ��!       J	N)���]r?N)���]r?!N)���]r?R      ��!       Z	N)���]r?N)���]r?!N)���]r?b      ��!       JGPUY]G'd�?b q���`�X@yB;�
=��?