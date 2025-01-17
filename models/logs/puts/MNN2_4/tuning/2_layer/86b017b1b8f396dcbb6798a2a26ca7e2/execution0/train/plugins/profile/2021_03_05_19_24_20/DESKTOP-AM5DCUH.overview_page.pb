�	��h o�7@��h o�7@!��h o�7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��h o�7@I�v|W4@1>x�҆��?AZd;�O��?Iy��M�@*	33333�I@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��ǘ���?!$��^�6?@)�!��u��?1����*;@:Preprocessing2U
Iterator::Model::ParallelMapV2�~j�t��?!7��<7@)�~j�t��?17��<7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��@��ǈ?!$��fP7@)vq�-�?1n��rq.@:Preprocessing2F
Iterator::Model46<��?!M|�wK�A@) �o_�y?1ƢfG(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�J�4q?!��DZ/ @)�J�4q?1��DZ/ @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�J�4�?!��DZ/P@)�q����o?1��E5�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!��DZ/@)�J�4a?1��DZ/@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���S㥋?!A��:@)Ǻ���V?1�W�Zx�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 86.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�12.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI>�!�b�X@Q���w^��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	I�v|W4@I�v|W4@!I�v|W4@      ��!       "	>x�҆��?>x�҆��?!>x�҆��?*      ��!       2	Zd;�O��?Zd;�O��?!Zd;�O��?:	y��M�@y��M�@!y��M�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q>�!�b�X@y���w^��?�"5
sequential/dense/MatMulMatMul�5�:ޡ�?!�5�:ޡ�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�h6�Ꮰ?!8�����?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�h6�Ꮰ?!����?"7
sequential/dense_1/MatMulMatMul�h6�Ꮰ?!�a�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�Is�M�?!���'�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�zڳ�?! #�D��?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�zڳ�?!k�cF�T�?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam�h6�Ꮠ?!�Mʅ�f�?"!
Adam/PowPow�h6�Ꮠ?!�1Ÿx�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�h6�Ꮠ?!��K�ZE�?Q      Y@Y>����/@aX�i��U@q�'L��mX@y����?"�
both�Your program is POTENTIALLY input-bound because 86.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�12.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 