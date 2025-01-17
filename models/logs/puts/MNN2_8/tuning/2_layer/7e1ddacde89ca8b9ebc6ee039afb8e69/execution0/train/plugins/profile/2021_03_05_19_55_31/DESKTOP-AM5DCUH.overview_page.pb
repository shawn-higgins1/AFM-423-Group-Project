�	�?�,-7@�?�,-7@!�?�,-7@	B�WF�r?B�WF�r?!B�WF�r?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�?�,-7@���E4@1V-���?A���Mb�?IO#-��@Y������P?*	      G@2U
Iterator::Model::ParallelMapV2�{�Pk�?!��,d!<@)�{�Pk�?1��,d!<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeattF��_�?!Nozӛ�9@)��ׁsF�?1�B���5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��_�L�?!�Mozӛ6@)9��v��z?1d!Y�B,@:Preprocessing2F
Iterator::Model�N@aÓ?!8��Mo�D@)-C��6z?1��Moz�+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!ozӛ�� @)�q����o?1ozӛ�� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipz6�>W�?!�B��M@)a��+ei?1ozӛ��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!���,d@)����Mb`?1���,d@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapg��j+��?!�7��Mo9@)��_�LU?1�Mozӛ@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9A�WF�r?Ie��{X@Q�2��/� @Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���E4@���E4@!���E4@      ��!       "	V-���?V-���?!V-���?*      ��!       2	���Mb�?���Mb�?!���Mb�?:	O#-��@O#-��@!O#-��@B      ��!       J	������P?������P?!������P?R      ��!       Z	������P?������P?!������P?b      ��!       JGPUYA�WF�r?b qe��{X@y�2��/� @�"5
sequential/dense/MatMulMatMul�0���ߢ?!�0���ߢ?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�C����?!^�E���?"C
%gradient_tape/sequential/dense/MatMulMatMul��{!Ǡ?!�g�R�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul%�廏[�?!�kW�8�?0"7
sequential/dense_1/MatMulMatMul%�廏[�?!|#�����?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchчB��?!v�3���?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMulчB��?!p�v�!�?"7
sequential/dense_2/MatMulMatMulчB��?!j�����?0"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam�l�F�Ɛ?!���"y��?"!
Adam/PowPow�l�F�Ɛ?!ʈ�E&y�?Q      Y@Y>����/@aX�i��U@q�b�V@yT������?"�
both�Your program is POTENTIALLY input-bound because 87.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�90.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 