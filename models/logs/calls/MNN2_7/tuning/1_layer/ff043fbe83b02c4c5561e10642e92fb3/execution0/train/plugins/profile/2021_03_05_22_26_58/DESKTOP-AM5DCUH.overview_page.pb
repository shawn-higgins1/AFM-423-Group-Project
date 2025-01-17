�	�?�:ss5@�?�:ss5@!�?�:ss5@	f<�\^"�?f<�\^"�?!f<�\^"�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�?�:ss5@��Y53@1֪]��?A���Mb�?I>>!;oc @Yq��#�?*����̌H@)       =2U
Iterator::Model::ParallelMapV2y�&1��?!�1��c�<@)y�&1��?1�1��c�<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�������?!�ٚ�Ou9@)'�����?1m
Ę��5@:Preprocessing2F
Iterator::Model^K�=��?!x�\j�|E@)�ZӼ�}?1��,~��,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate'�����?!m
Ę��5@)y�&1�|?1�1��c�,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vOn?!���7$@)���_vOn?1���7$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipy�&1��?!�1��c�L@)��H�}m?1|�f�S@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!|�f�S@)��H�}]?1|�f�S@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaptF��_�?!��{<8@)a2U0*�S?1d�ΙK�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9e<�\^"�?I�}��Y�X@Qy�q B�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��Y53@��Y53@!��Y53@      ��!       "	֪]��?֪]��?!֪]��?*      ��!       2	���Mb�?���Mb�?!���Mb�?:	>>!;oc @>>!;oc @!>>!;oc @B      ��!       J	q��#�?q��#�?!q��#�?R      ��!       Z	q��#�?q��#�?!q��#�?b      ��!       JGPUYe<�\^"�?b q�}��Y�X@yy�q B�?�"5
sequential/dense/MatMulMatMul3X�턦?!3X�턦?0"C
%gradient_tape/sequential/dense/MatMulMatMul��5�`�?!����D�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��2y�?!#�)Q��?"7
sequential/dense_1/MatMulMatMul��2y�?!'B�α��?0"!
Adam/PowPow��5�`�?!����=d�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch��5�`�?!�2���?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad��5�`�?!�_�*Ve�?"E
'gradient_tape/sequential/dense_1/MatMulMatMul����=d�?!bފ����?0"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam�u�o��?!���F��?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam�u�o��?!����?Q      Y@Y��/Ċ�0@a�	�N]�T@q-�P�:V@y��Z�!u�?"�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�88.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 