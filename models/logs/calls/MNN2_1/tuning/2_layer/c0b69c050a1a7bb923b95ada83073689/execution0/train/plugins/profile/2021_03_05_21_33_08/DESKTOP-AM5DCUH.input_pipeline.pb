	B碫򁫥@B碫򁫥@!B碫򁫥@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-B碫򁫥@x%蓅}�4@1^帜_严?A鄿ソ璃?Iq猥�9@*	翁烫虒H@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate@a脫?!穎馭]@)郸y�):�?1�:韐S B@:Preprocessing2U
Iterator::Model::ParallelMapV2硔虬Pk�?!$畺軪:@)硔虬Pk�?1$畺軪:@:Preprocessing2F
Iterator::Model+�傥鲹?!Z96葊跜@)F%u�{?1j渱驡�*@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��0�*x?!w箺W(@)孞�4q?1��&�!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip豞vO�?!ζ�7$N@)a糜+ei?1MVQ,A@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_�Q赱?!從詹@)_�Q赱?1從詹@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�0�*�?!呖扒C鶧@)旜_楲U?1倰�;g.@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor捤H縸M?!|秄馭�?)捤H縸M?1|秄馭�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*〤?!d橩嶓?)a2U0*〤?1d橩嶓?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI<圍c絏@Q1^Ш�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	x%蓅}�4@x%蓅}�4@!x%蓅}�4@      ��!       "	^帜_严?^帜_严?!^帜_严?*      ��!       2	鄿ソ璃?鄿ソ璃?!鄿ソ璃?:	q猥�9@q猥�9@!q猥�9@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q<圍c絏@y1^Ш�?