�	:�����3@:�����3@!:�����3@	�Yd)w��?�Yd)w��?!�Yd)w��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6:�����3@�D��i1@1+��B��?A�=yX��?I�m��?Y.����w?*	������J@2U
Iterator::Model::ParallelMapV2����Mb�?!���=@)����Mb�?1���=@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap� �	��?!c"=P9�<@)46<�R�?1��[�U4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��@��ǈ?!sv��6@)M�O��?1T�n��2@:Preprocessing2F
Iterator::Model�z6�>�?!���@�,E@)S�!�uq{?1      )@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice;�O��nr?!��� @);�O��nr?1��� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��?��?!sv��L@)_�Q�k?1=P9��_@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!���@)����Mb`?1���@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�Yd)w��?I�a�X@Q�b���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�D��i1@�D��i1@!�D��i1@      ��!       "	+��B��?+��B��?!+��B��?*      ��!       2	�=yX��?�=yX��?!�=yX��?:	�m��?�m��?!�m��?B      ��!       J	.����w?.����w?!.����w?R      ��!       Z	.����w?.����w?!.����w?b      ��!       JGPUY�Yd)w��?b q�a�X@y�b���?�"5
sequential/dense/MatMulMatMul��ϢV��?!��ϢV��?0"C
%gradient_tape/sequential/dense/MatMulMatMul.w<%=��?!X�ɤ�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�4|��?!Ep��bP�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad;*�ɾ��?!�������?"7
sequential/dense_1/MatMulMatMul;*�ɾ��?!�z�s���?0"
Sum_4Sumr��[�?!)mH
��?"E
'gradient_tape/sequential/dense_1/MatMulMatMul@�(�%Z�?!DrO��?0"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam�5�)�?!t,s���?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�5�)�?!ƟdY��?"!
Adam/PowPow�5�)�?!�V���?Q      Y@Y�M�_{4@a��(�S@q��� fvW@y� ����?"�
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
Refer to the TF2 Profiler FAQb�93.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 