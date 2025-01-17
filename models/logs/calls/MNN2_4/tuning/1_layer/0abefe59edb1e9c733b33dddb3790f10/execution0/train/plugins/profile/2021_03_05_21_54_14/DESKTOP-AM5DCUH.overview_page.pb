�	�!S>m6@�!S>m6@!�!S>m6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�!S>m6@D�����3@1�#EdX�?A�H.�!��?I���@*	33333�K@2U
Iterator::Model::ParallelMapV2%u��?!���~L:@)%u��?1���~L:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���Q��?!z|�h�:@)�������?1���\6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�<,Ԛ�?!%���9:@)'�����?1g���-$3@:Preprocessing2F
Iterator::Model��A�f�?!�ߙP��B@)a��+ey?1Q��D�.&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�Q���?! f�ONO@)	�^)�p?1�����V@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!����/�@)�q����o?1����/�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{�G�zd?!R�?��@){�G�zd?1R�?��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���H�?!��_�q<@)��_�LU?1��)kʚ@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�!�5�X@Q4����8�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	D�����3@D�����3@!D�����3@      ��!       "	�#EdX�?�#EdX�?!�#EdX�?*      ��!       2	�H.�!��?�H.�!��?!�H.�!��?:	���@���@!���@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�!�5�X@y4����8�?�"5
sequential/dense/MatMulMatMulW�w`b��?!W�w`b��?0"C
%gradient_tape/sequential/dense/MatMulMatMul�L�#�?!��>2e�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulRI����?!�cQ�Լ?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�_��+�?!������?"7
sequential/dense_1/MatMulMatMul�_��+�?!�-����?0"!
Adam/PowPow��LD$�?!<M�u�9�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�L�#�?!��h���?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�L�#�?!l`�HB�?"*

LogicalAnd
LogicalAnd���7�?!VA�u�%�?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdamsJ+�4�?!D�8Ԁ�?Q      Y@Y��/Ċ�0@a�	�N]�T@qzV:��gX@ys�����?"�
both�Your program is POTENTIALLY input-bound because 87.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 