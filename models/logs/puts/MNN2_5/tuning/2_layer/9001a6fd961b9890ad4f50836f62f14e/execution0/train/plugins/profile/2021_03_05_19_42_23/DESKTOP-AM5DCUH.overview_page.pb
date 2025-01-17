�	oH�'�6@oH�'�6@!oH�'�6@	�O���?�O���?!�O���?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6oH�'�6@8��+��3@1���{G��?A���Mb�?IT����C@Y��CV�?*	�����I@2U
Iterator::Model::ParallelMapV2Ǻ����?!�|\?�Z6@)Ǻ����?1�|\?�Z6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�{�Pk�?!Q=h䝿9@)��_vO�?1����5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�Pk�w�?!*6~E��;@)�j+��݃?1i���|\3@:Preprocessing2F
Iterator::Model������?!
v��A@)��0�*x?1�ޜy��'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipL7�A`�?!�D���wP@)/n��r?1@�ZV��!@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�J�4q?!�]�/7� @)�J�4q?1�]�/7� @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!�]�/7�@)�J�4a?1�]�/7�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap2U0*��?!}\?�ZV?@)��H�}]?1�2	v�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�O���?I��Qb�X@Q�n,�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	8��+��3@8��+��3@!8��+��3@      ��!       "	���{G��?���{G��?!���{G��?*      ��!       2	���Mb�?���Mb�?!���Mb�?:	T����C@T����C@!T����C@B      ��!       J	��CV�?��CV�?!��CV�?R      ��!       Z	��CV�?��CV�?!��CV�?b      ��!       JGPUY�O���?b q��Qb�X@y�n,�?�"5
sequential/dense/MatMulMatMul`��v$�?!`��v$�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�'� ��?!V�n{�?0"7
sequential/dense_1/MatMulMatMulV ����?!3�N_w��?0"E
'gradient_tape/sequential/dense_1/MatMulMatMulH�	ȝ?!���;�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�x���Ɲ?!�)��?"7
sequential/dense_2/MatMulMatMulk�%��D�?!"�"�e�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch+d��"�?!gRO�
�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�e�	�ޔ?!_�c��?"!
Adam/PowPowV ����?!'è_q��?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradV ����?!��-xs�?Q      Y@Y>����/@aX�i��U@q$���ȔV@y�boO3��?"�
both�Your program is POTENTIALLY input-bound because 87.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�90.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 