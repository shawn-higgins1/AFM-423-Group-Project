�	ԛQ��6@ԛQ��6@!ԛQ��6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-ԛQ��6@,H3Mw4@1���s��?A$����ۧ?I���˚8@*	������G@2U
Iterator::Model::ParallelMapV2�]K�=�?!������;@)�]K�=�?1������;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��@��ǈ?!"�k"�k9@)��ZӼ�?1۶m۶m5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��0�*�?!�7��7�8@)vq�-�?1O��O��0@:Preprocessing2F
Iterator::Modela2U0*��?!t+t+D@)��0�*x?1�7��7�(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!>�b>�b @)�q����o?1>�b>�b @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�ZӼ��?!��ԋ��M@)-C��6j?1���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!7��7��@)ŏ1w-!_?17��7��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��<,Ԋ?!�`��`�;@)��_�LU?1�F��F�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI>�q���X@Qp���\�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	,H3Mw4@,H3Mw4@!,H3Mw4@      ��!       "	���s��?���s��?!���s��?*      ��!       2	$����ۧ?$����ۧ?!$����ۧ?:	���˚8@���˚8@!���˚8@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q>�q���X@yp���\�?�"5
sequential/dense/MatMulMatMul'B�z���?!'B�z���?0"C
%gradient_tape/sequential/dense/MatMulMatMulx��ޠ?!����?0"7
sequential/dense_1/MatMulMatMulx��ޠ?!���Th\�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul��,0���?!q�6�P�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul.��,Hz�?!P�Ano�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch֞�ݹ�?!+Yo}E�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad֞�ݹ�?!� ���?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul֞�ݹ�?!�@���W�?"7
sequential/dense_2/MatMulMatMul֞�ݹ�?!���0���?0"!
Adam/PowPowx��ސ?!��IU�?Q      Y@Y>����/@aX�i��U@q4��{X@y���Th\�?"�
both�Your program is POTENTIALLY input-bound because 89.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 