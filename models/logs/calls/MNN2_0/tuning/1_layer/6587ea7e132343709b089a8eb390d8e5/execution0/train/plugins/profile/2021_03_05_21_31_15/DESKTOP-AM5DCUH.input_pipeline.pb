	適􂩝@適􂩝@!適􂩝@	$	�鲹?$	�鲹?!$	�鲹?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6適􂩝@% ��3@1]o洨徥?A淢G 7嫥?I�+茯� @Y/鯺|q?*	ffffffK@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�0�*�?!袂B@)垸~j紅�?1僯仙鬠A@:Preprocessing2U
Iterator::Model::ParallelMapV2X9慈v緩?!拳�;郒<@)X9慈v緩?1拳�;郒<@:Preprocessing2F
Iterator::Model
祝p=
�?!"uy嘍@)y�&1瑋?1鳒L_%�)@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�<,詺鎪?!渲�p�*@)篒+噕?1T{NΟ$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipaTR'爥�?!�G妴xM@)a糜+ei?1覹	c碃@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor捤H縸]?!@n]�G
@)捤H縸]?1@n]�G
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapw-!鬺�?!+alT鸆@)旜_楲U?1fA电d�@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor捤H縸M?!@n]�G�?)捤H縸M?1@n]�G�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice呛笉餏?!皱p�?)呛笉餏?1皱p�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9#	�鲹?IbE芇耎@Qp�%�S�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	% ��3@% ��3@!% ��3@      ��!       "	]o洨徥?]o洨徥?!]o洨徥?*      ��!       2	淢G 7嫥?淢G 7嫥?!淢G 7嫥?:	�+茯� @�+茯� @!�+茯� @B      ��!       J	/鯺|q?/鯺|q?!/鯺|q?R      ��!       Z	/鯺|q?/鯺|q?!/鯺|q?b      ��!       JGPUY#	�鲹?b qbE芇耎@yp�%�S�?